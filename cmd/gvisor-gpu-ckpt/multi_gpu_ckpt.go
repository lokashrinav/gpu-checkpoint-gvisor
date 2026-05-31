//go:build linux

// Save-restore-exec helper for multi-GPU checkpoint/restore.
// Save: lock GPUs (cuda-checkpoint) + signal app to DMA copy + destroy contexts.
// Restore: signal app to re-init CUDA from host mirrors.
package main

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

typedef int (*CkptFn)(int, void*);
typedef int (*InitFn)(unsigned int);

static int gpu_lock() {
	void *h = dlopen("libcuda.so.1", RTLD_NOW);
	if (!h) { fprintf(stderr, "gpu-ckpt: dlopen: %s\n", dlerror()); return -1; }
	InitFn init = (InitFn)dlsym(h, "cuInit");
	CkptFn lock = (CkptFn)dlsym(h, "cuCheckpointProcessLock");
	if (!init || !lock) { fprintf(stderr, "gpu-ckpt: symbols missing\n"); return -2; }
	int r = init(0);
	if (r != 0) { fprintf(stderr, "gpu-ckpt: cuInit=%d\n", r); return r; }
	char args[64]; memset(args, 0, 64);
	r = lock(getpid(), args);
	fprintf(stderr, "gpu-ckpt: lock=%d\n", r);
	return r;
}
*/
import "C"
import (
	"fmt"
	"os"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

var _ = unsafe.Pointer(nil)

func waitForFlag() error {
	for i := 0; i < 60; i++ {
		if d, err := os.ReadFile("/tmp/.gpu_ckpt_done"); err == nil {
			rc := strings.TrimSpace(string(d))
			if rc != "0" {
				return fmt.Errorf("app returned rc=%s", rc)
			}
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for app")
}

func main() {
	mode := os.Getenv("GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE")
	if mode == "" {
		fmt.Fprintln(os.Stderr, "gpu-ckpt: MODE not set")
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "gpu-ckpt: mode=%s\n", mode)

	switch mode {
	case "save":
		// Step 1: Lock all GPUs atomically via cuda-checkpoint
		rc := C.gpu_lock()
		if rc != 0 && rc != 3 {
			fmt.Fprintf(os.Stderr, "gpu-ckpt: lock failed rc=%d\n", rc)
			os.Exit(1)
		}
		if rc == 0 {
			fmt.Fprintln(os.Stderr, "gpu-ckpt: GPUs locked")
		}

		// Step 2: Signal app to DMA copy GPU→host + destroy contexts
		os.Remove("/tmp/.gpu_ckpt_done")
		if err := syscall.Kill(1, syscall.SIGUSR1); err != nil {
			fmt.Fprintf(os.Stderr, "gpu-ckpt: signal save: %v\n", err)
			os.Exit(1)
		}
		if err := waitForFlag(); err != nil {
			fmt.Fprintf(os.Stderr, "gpu-ckpt: save: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "gpu-ckpt: save complete")

	case "restore", "resume":
		// Signal app to re-init CUDA from host mirrors
		os.Remove("/tmp/.gpu_ckpt_done")
		if err := syscall.Kill(1, syscall.SIGUSR2); err != nil {
			fmt.Fprintf(os.Stderr, "gpu-ckpt: signal restore: %v\n", err)
			os.Exit(1)
		}
		if err := waitForFlag(); err != nil {
			fmt.Fprintf(os.Stderr, "gpu-ckpt: restore: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "gpu-ckpt: restore complete")

	default:
		fmt.Fprintf(os.Stderr, "gpu-ckpt: unknown mode %q\n", mode)
		os.Exit(1)
	}
}
