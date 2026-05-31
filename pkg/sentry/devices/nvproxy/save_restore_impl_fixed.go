// Copyright 2023 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !false
// +build !false

package nvproxy

import (
	goContext "context"
	"fmt"
	"path/filepath"

	"golang.org/x/sys/unix"
	"unsafe"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/devutil"
	"gvisor.dev/gvisor/pkg/fdnotifier"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/waiter"
)

type restoreContext struct {
	context.NoTask
	log.Logger
	goContext.Context
}

func newRestoreContext(ctx goContext.Context) context.Context {
	return &restoreContext{Context: ctx, Logger: log.Log()}
}

func (nvp *nvproxy) beforeSaveImpl() {
	// Keep client/object state for RM replay on restore.
}
func (nvp *nvproxy) afterLoadImpl(goContext.Context) {
	// no-op: frontendFDs map is restored by stateify,
	// ABI is rebuilt in save_restore.go:afterLoad().
}

func (fd *frontendFD) beforeSaveImpl() {
	// hostFD value is serialized but stale on restore; afterLoadImpl
	// replaces it with a freshly opened FD.
}

func (fd *frontendFD) afterLoadImpl(goCtx goContext.Context) {
	ctx := newRestoreContext(goCtx)
	devPath := fd.dev.basename()

	// Reopen host device file.
	if fd.dev.nvp.useDevGofer {
		provider := devutil.GoferClientProviderFromContext(ctx)
		if provider == nil {
			panic(fmt.Sprintf("nvproxy: device gofer client provider not available for %s on restore", devPath))
		}
		devClient := provider.GetDevGoferClient(fd.containerName)
		if devClient == nil {
			panic(fmt.Sprintf("nvproxy: device gofer client for container %q not available on restore", fd.containerName))
		}
		hostFD, err := devClient.OpenAt(ctx, devPath, unix.O_RDWR)
		if err != nil {
			panic(fmt.Sprintf("nvproxy: failed to reopen device %s via gofer on restore: %v", devPath, err))
		}
		fd.hostFD = int32(hostFD)
	} else {
		abspath := filepath.Join("/dev", devPath)
		hostFD, err := unix.Openat(-1, abspath, unix.O_RDWR|unix.O_NOFOLLOW, 0)
		if err != nil {
			panic(fmt.Sprintf("nvproxy: failed to reopen host %s on restore: %v", abspath, err))
		}
		fd.hostFD = int32(hostFD)
	}

	// Re-initialize the eventListener callback (not serialized) but do NOT
	// re-register the entry; it's already in the queue from checkpoint state.
	fd.internalEntry.Init(fd, waiter.AllEvents)
	if err := fdnotifier.AddFD(fd.hostFD, &fd.internalQueue); err != nil {
		panic(fmt.Sprintf("nvproxy: fdnotifier.AddFD failed on restore: %v", err))
	}

	fd.memmapFile.SetFD(int(fd.hostFD))

	// Replay RM object tree on the new host FD: root client -> device -> subdevice.
	// This re-establishes device bindings so cuCtxCreate works after restore.
	fd.dev.nvp.clientsMu.RLock()
	ioctlCmd := uintptr(linux.IOWR(nvgpu.NV_IOCTL_MAGIC, nvgpu.NV_ESC_RM_ALLOC, nvgpu.SizeofNVOS64Parameters))
	for _, client := range fd.dev.nvp.clients {
		// Step 1: Replay root client
		params := client.params.ioctlParams
		params.Status = 0
		params.PAllocParms = 0
		params.PRightsRequested = 0
		_, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd.hostFD), ioctlCmd, uintptr(unsafe.Pointer(&params)))
		if errno != 0 {
			continue
		}
		log.Infof("nvproxy: replayed root client %v", client.handle)

		// Step 2: Replay device and subdevice objects
		classOrder := []nvgpu.ClassID{nvgpu.NV01_DEVICE_0, nvgpu.NV20_SUBDEVICE_0}
		for _, targetClass := range classOrder {
			client.objsMu.Lock()
			for _, obj := range client.resources {
				if obj.class == targetClass {
					if rmObj, ok := obj.impl.(*rmAllocObject); ok {
						p := rmObj.params.ioctlParams
						p.Status = 0
						if len(rmObj.params.allocParams) > 0 {
							p.PAllocParms = p64FromPtr(unsafe.Pointer(&rmObj.params.allocParams[0]))
						} else {
							p.PAllocParms = 0
						}
						p.PRightsRequested = 0
						_, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd.hostFD), ioctlCmd, uintptr(unsafe.Pointer(&p)))
						if errno != 0 {
							log.Warningf("nvproxy: replay %v (class %#x) failed: %v", obj.handle, targetClass, errno)
						} else {
							log.Infof("nvproxy: replayed %v (class %#x)", obj.handle, targetClass)
						}
					}
				}
			}
			client.objsMu.Unlock()
		}
	}
	fd.dev.nvp.clientsMu.RUnlock()
}

func (fd *uvmFD) beforeSaveImpl() {
	// hostFD value is serialized but stale on restore; afterLoadImpl
	// replaces it with a freshly opened FD.
}

func (fd *uvmFD) afterLoadImpl(goCtx goContext.Context) {
	ctx := newRestoreContext(goCtx)
	// Reopen host device file.
	if fd.dev.nvp.useDevGofer {
		provider := devutil.GoferClientProviderFromContext(ctx)
		if provider == nil {
			panic("nvproxy: device gofer client provider not available for nvidia-uvm on restore")
		}
		devClient := provider.GetDevGoferClient(fd.containerName)
		if devClient == nil {
			panic(fmt.Sprintf("nvproxy: device gofer client for container %q not available on restore", fd.containerName))
		}
		hostFD, err := devClient.OpenAt(ctx, "nvidia-uvm", unix.O_RDWR)
		if err != nil {
			panic(fmt.Sprintf("nvproxy: failed to reopen nvidia-uvm via gofer on restore: %v", err))
		}
		fd.hostFD = int32(hostFD)
	} else {
		hostFD, err := unix.Openat(-1, "/dev/nvidia-uvm", unix.O_RDWR|unix.O_NOFOLLOW, 0)
		if err != nil {
			panic(fmt.Sprintf("nvproxy: failed to reopen host /dev/nvidia-uvm on restore: %v", err))
		}
		fd.hostFD = int32(hostFD)
	}

	if err := fdnotifier.AddFD(fd.hostFD, &fd.queue); err != nil {
		panic(fmt.Sprintf("nvproxy: fdnotifier.AddFD failed on restore: %v", err))
	}

	fd.memmapFile.SetFD(int(fd.hostFD))
	fd.memmapFile.RequireAddrEqualsFileOffset()
}
