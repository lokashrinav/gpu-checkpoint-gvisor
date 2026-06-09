//go:build linux

// Binary gvisor-gpu-ckpt is a SaveRestoreExec binary for gVisor that
// handles GPU checkpoint/restore via NVIDIA's cuCheckpointProcess* API.
//
// This binary runs INSIDE the gVisor sandbox. It targets the container's
// init process (PID 1) because cuda-checkpoint ioctls must go through
// nvproxy to reach the real NVIDIA driver. Calling from the host with
// the sentry PID fails because the sentry creates GPU contexts via raw
// ioctls without libcuda.so initialization.
//
// gVisor sets GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE ("save", "restore", "resume").
// GPU checkpoint mode is passed as argv[1] via --save-restore-exec-argv,
// or via GVISOR_GPU_CHECKPOINT_MODE env var, defaulting to "direct".
//
// Modes:
//
//	direct:  Helper calls cuCheckpointProcess* directly on PID 1.
//	         Works for single/multi-GPU without NCCL.
//	signal:  Helper sends SIGUSR1 (save) / SIGUSR2 (restore) to PID 1.
//	         App must register handlers that call cuCheckpointProcess*
//	         and ncclCommSuspend/Resume. Required for NCCL workloads.
//	hybrid:  Helper calls cuCheckpointProcessLock (atomic GPU freeze),
//	         then signals PID 1 for app to checkpoint/restore.
//	         Best for multi-GPU + NCCL.
//
// Usage with gVisor:
//
//	runsc checkpoint --save-restore-exec-argv=/path/to/gvisor-gpu-ckpt <container-id>
//	runsc checkpoint --gpu-checkpoint=/path/to/gvisor-gpu-ckpt <container-id>
//	runsc checkpoint --gpu-checkpoint=signal:/path/to/gvisor-gpu-ckpt <container-id>
//	runsc checkpoint --gpu-checkpoint=hybrid:/path/to/gvisor-gpu-ckpt <container-id>
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const completionFile = "/tmp/.gpu_ckpt_done"

func main() {
	mode := os.Getenv("GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE")
	if mode == "" {
		fmt.Fprintln(os.Stderr, "gvisor-gpu-ckpt: GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE not set")
		os.Exit(1)
	}

	// GPU mode: argv[1] takes precedence (passed via --save-restore-exec-argv),
	// then env var, then default to direct.
	gpuMode := "direct"
	if len(os.Args) > 1 {
		gpuMode = os.Args[1]
	} else if env := os.Getenv("GVISOR_GPU_CHECKPOINT_MODE"); env != "" {
		gpuMode = env
	}

	pid, err := getTargetPID()
	if err != nil {
		fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: failed to determine target PID: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: mode=%s gpu_mode=%s pid=%d\n", mode, gpuMode, pid)

	switch gpuMode {
	case "direct":
		if err := runDirect(mode, pid); err != nil {
			fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: %s failed: %v\n", mode, err)
			os.Exit(1)
		}
	case "signal":
		if err := runSignal(mode, pid); err != nil {
			fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: %s failed: %v\n", mode, err)
			os.Exit(1)
		}
	case "hybrid":
		if err := runHybrid(mode, pid); err != nil {
			fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: %s failed: %v\n", mode, err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: unknown gpu mode %q\n", gpuMode)
		os.Exit(1)
	}
}

// runDirect handles direct mode: helper calls cuCheckpointProcess* directly.
func runDirect(mode string, pid int) error {
	if err := loadLibcuda(); err != nil {
		fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: %v (no GPU contexts to checkpoint)\n", err)
		return nil
	}

	switch mode {
	case "save":
		return doDirectSave(pid)
	case "restore", "resume":
		return doDirectRestore(pid)
	default:
		return fmt.Errorf("unknown mode %q", mode)
	}
}

// runSignal handles signal mode: sends SIGUSR1/SIGUSR2 to PID 1, app
// handles everything (ncclCommSuspend + cuCheckpointProcess* + ncclCommResume).
func runSignal(mode string, pid int) error {
	os.Remove(completionFile)

	var sig syscall.Signal
	switch mode {
	case "save":
		sig = syscall.SIGUSR1
	case "restore", "resume":
		sig = syscall.SIGUSR2
	default:
		return fmt.Errorf("unknown mode %q", mode)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: sending signal %d to PID %d\n", sig, pid)
	if err := syscall.Kill(pid, sig); err != nil {
		return fmt.Errorf("kill(%d, %d): %w", pid, sig, err)
	}

	return waitForCompletion(30 * time.Second)
}

// runHybrid handles hybrid mode: helper locks GPUs, then signals app for
// checkpoint/restore. Combines atomic GPU locking with app-side NCCL handling.
func runHybrid(mode string, pid int) error {
	if err := loadLibcuda(); err != nil {
		fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: %v (no GPU contexts to checkpoint)\n", err)
		return nil
	}

	switch mode {
	case "save":
		return doHybridSave(pid)
	case "restore", "resume":
		return doHybridRestore(pid)
	default:
		return fmt.Errorf("unknown mode %q", mode)
	}
}

func doDirectSave(pid int) error {
	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: locking GPU contexts for PID %d\n", pid)
	if err := checkpointLock(pid); err != nil {
		if isNoContextError(err) {
			fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: no CUDA contexts for PID %d, nothing to checkpoint\n", pid)
			return nil
		}
		return fmt.Errorf("lock: %w", err)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: checkpointing GPU state for PID %d\n", pid)
	if err := checkpointCheckpoint(pid); err != nil {
		_ = checkpointUnlock(pid)
		return fmt.Errorf("checkpoint: %w", err)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: GPU checkpoint complete for PID %d\n", pid)
	return nil
}

func doDirectRestore(pid int) error {
	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: restoring GPU state for PID %d\n", pid)
	if err := checkpointRestore(pid); err != nil {
		return fmt.Errorf("restore: %w", err)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: unlocking GPU contexts for PID %d\n", pid)
	if err := checkpointUnlock(pid); err != nil {
		return fmt.Errorf("unlock: %w", err)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: GPU restore complete for PID %d\n", pid)
	return nil
}

func doHybridSave(pid int) error {
	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: locking GPU contexts for PID %d (hybrid)\n", pid)
	if err := checkpointLock(pid); err != nil {
		if isNoContextError(err) {
			fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: no CUDA contexts for PID %d, nothing to checkpoint\n", pid)
			return nil
		}
		return fmt.Errorf("lock: %w", err)
	}

	os.Remove(completionFile)
	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: GPUs locked, sending SIGUSR1 to PID %d for app checkpoint\n", pid)
	if err := syscall.Kill(pid, syscall.SIGUSR1); err != nil {
		_ = checkpointUnlock(pid)
		return fmt.Errorf("kill(%d, SIGUSR1): %w", pid, err)
	}

	if err := waitForCompletion(30 * time.Second); err != nil {
		_ = checkpointUnlock(pid)
		return fmt.Errorf("waiting for app checkpoint: %w", err)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: hybrid checkpoint complete for PID %d\n", pid)
	return nil
}

func doHybridRestore(pid int) error {
	os.Remove(completionFile)
	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: sending SIGUSR2 to PID %d for app restore\n", pid)
	if err := syscall.Kill(pid, syscall.SIGUSR2); err != nil {
		return fmt.Errorf("kill(%d, SIGUSR2): %w", pid, err)
	}

	if err := waitForCompletion(30 * time.Second); err != nil {
		return fmt.Errorf("waiting for app restore: %w", err)
	}

	fmt.Fprintf(os.Stderr, "gvisor-gpu-ckpt: hybrid restore complete for PID %d\n", pid)
	return nil
}

func waitForCompletion(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(completionFile); err == nil {
			os.Remove(completionFile)
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for %s (after %v)", completionFile, timeout)
}

func isNoContextError(err error) bool {
	return err != nil && strings.Contains(err.Error(), "rc=3")
}

func getTargetPID() (int, error) {
	if s := os.Getenv("GVISOR_CHECKPOINT_PID"); s != "" {
		pid, err := strconv.Atoi(s)
		if err != nil {
			return 0, fmt.Errorf("invalid GVISOR_CHECKPOINT_PID %q: %w", s, err)
		}
		return pid, nil
	}
	return 1, nil
}
