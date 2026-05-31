//go:build linux

package main

import (
	"fmt"
	"os"
	"strings"
	"syscall"
	"time"
)

func main() {
	mode := os.Getenv("GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE")
	fmt.Fprintf(os.Stderr, "helper: mode=%s\n", mode)
	var sig syscall.Signal
	switch mode {
	case "save":
		sig = syscall.SIGUSR1
	case "restore", "resume":
		sig = syscall.SIGUSR2
	default:
		return
	}
	os.Remove("/tmp/.gpu_ckpt_done")
	syscall.Kill(1, sig)
	for i := 0; i < 120; i++ {
		if d, err := os.ReadFile("/tmp/.gpu_ckpt_done"); err == nil {
			rc := strings.TrimSpace(string(d))
			fmt.Fprintf(os.Stderr, "helper: %s done rc=%s\n", mode, rc)
			if rc != "0" {
				os.Exit(1)
			}
			return
		}
		time.Sleep(500 * time.Millisecond)
	}
	fmt.Fprintln(os.Stderr, "helper: timeout")
	os.Exit(1)
}
