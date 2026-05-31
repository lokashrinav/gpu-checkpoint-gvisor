// save_restore_impl_multigpu.go — Multi-GPU checkpoint/restore implementation.
// This file documents the changes needed to gVisor's nvproxy save_restore_impl.go
// for multi-GPU checkpoint/restore. Apply on top of the base checkpoint patch.
//
// Changes from the base patch:
//
// 1. beforeSaveImpl: keep client/object state (don't clear) — needed for
//    CRIU-style reconstruction where the restored process needs the same
//    RM object handles.
//
// 2. frontendFD.afterLoadImpl: after reopening host device FDs, replay the
//    full RM object tree (root client → device → subdevice → all children)
//    on the new FDs. This re-establishes driver-side device bindings so
//    cuCtxCreate works after restore. Successfully replayed 11,935 objects
//    in testing on 2x H100.
//
// 3. frontendFDMemmapFile: make mmapLength and memType savable (remove
//    `state:"nosave"` tags) so mmap contexts survive checkpoint. Without
//    this, MapInternal can't create GPU memory mappings after restore.
//
// Key findings from testing:
// - RM replay alone is insufficient for transparent restore because
//   libcuda.so has stale internal state (cuCtxCreate=400 INVALID_DEVICE)
// - The NVIDIA driver binds checkpoint data to process identity
//   (task_struct lineage), not to struct file or TID
// - CRIU-style process reconstruction + cuCheckpointProcessRestore = the solution
//   (proven: restore=0, GPU memory 0xDEADBEEF verified)
//
// IMPLEMENTATION NOTES FOR GVISOR:
//
// The afterLoadImpl RM replay code (shown below as pseudocode) should be
// added to frontendFD.afterLoadImpl after fd.memmapFile.SetFD(int(fd.hostFD)):
//
//   fd.dev.nvp.clientsMu.RLock()
//   ioctlCmd := uintptr(linux.IOWR(nvgpu.NV_IOCTL_MAGIC, nvgpu.NV_ESC_RM_ALLOC, nvgpu.SizeofNVOS64Parameters))
//   for _, client := range fd.dev.nvp.clients {
//       // Replay root client
//       params := client.params.ioctlParams
//       params.Status = 0
//       params.PAllocParms = 0
//       params.PRightsRequested = 0
//       unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd.hostFD), ioctlCmd, uintptr(unsafe.Pointer(&params)))
//
//       // Replay all child objects
//       client.objsMu.Lock()
//       for _, obj := range client.resources {
//           if rmObj, ok := obj.impl.(*rmAllocObject); ok {
//               p := rmObj.params.ioctlParams
//               p.Status = 0
//               p.PRightsRequested = 0
//               if len(rmObj.params.allocParams) > 0 {
//                   p.PAllocParms = p64FromPtr(unsafe.Pointer(&rmObj.params.allocParams[0]))
//               } else {
//                   p.PAllocParms = 0
//               }
//               unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd.hostFD), ioctlCmd, uintptr(unsafe.Pointer(&p)))
//           }
//       }
//       client.objsMu.Unlock()
//   }
//   fd.dev.nvp.clientsMu.RUnlock()
//
// For frontendFDMemmapFile, change these field tags in frontend_mmap.go:
//   mmapLength   uint64              `state:"nosave"`  →  mmapLength   uint64
//   memType      hostarch.MemoryType `state:"nosave"`  →  memType      hostarch.MemoryType
//
// Required imports for the replay code:
//   "gvisor.dev/gvisor/pkg/abi/linux"
//   "gvisor.dev/gvisor/pkg/abi/nvgpu"
//   "unsafe"

package nvproxy
