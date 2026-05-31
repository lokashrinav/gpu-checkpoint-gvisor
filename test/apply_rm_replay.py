#!/usr/bin/env python3
"""Add RM client replay to frontendFD.afterLoadImpl in save_restore_impl.go"""

path = "/tmp/gvisor/pkg/sentry/devices/nvproxy/save_restore_impl.go"
with open(path) as f:
    code = f.read()

# Revert beforeSaveImpl to keep state for replay
import re
m = re.search(r"func \(nvp \*nvproxy\) beforeSaveImpl\(\) \{[^}]+\}", code, re.DOTALL)
if m:
    code = code.replace(m.group(),
        "func (nvp *nvproxy) beforeSaveImpl() {\n"
        "\t// Keep client/object state for RM replay on restore.\n"
        "}")
    print("Reverted beforeSaveImpl")

# Find the FIRST occurrence of fd.memmapFile.SetFD (frontendFD, not uvmFD)
target = "\tfd.memmapFile.SetFD(int(fd.hostFD))\n}"
idx = code.index(target)

# Replace with SetFD + RM replay
replacement = '''\tfd.memmapFile.SetFD(int(fd.hostFD))

\t// Replay RM root client allocation on the new host FD so the CUDA
\t// library's cached handles match the driver's session state.
\tfd.dev.nvp.clientsMu.RLock()
\tfor _, client := range fd.dev.nvp.clients {
\t\tif client.params.fd == fd {
\t\t\tparams := client.params.ioctlParams
\t\t\tparams.Status = 0
\t\t\tparams.PAllocParms = 0
\t\t\tparams.PRightsRequested = 0
\t\t\tioctlCmd := uintptr(linux.IOWR(nvgpu.NV_IOCTL_MAGIC, nvgpu.NV_ESC_RM_ALLOC, nvgpu.SizeofNVOS64Parameters))
\t\t\t_, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd.hostFD), ioctlCmd, uintptr(unsafe.Pointer(&params)))
\t\t\tif errno != 0 {
\t\t\t\tlog.Warningf("nvproxy: RM client replay failed: %v", errno)
\t\t\t} else {
\t\t\t\tlog.Infof("nvproxy: replayed RM client %v on restored FD", client.handle)
\t\t\t}
\t\t}
\t}
\tfd.dev.nvp.clientsMu.RUnlock()
}'''

code = code[:idx] + replacement + code[idx+len(target):]

# Add imports
if '"gvisor.dev/gvisor/pkg/abi/linux"' not in code:
    code = code.replace(
        '"gvisor.dev/gvisor/pkg/abi/nvgpu"',
        '"gvisor.dev/gvisor/pkg/abi/linux"\n\t"gvisor.dev/gvisor/pkg/abi/nvgpu"')
if '"unsafe"' not in code:
    code = code.replace(
        '"golang.org/x/sys/unix"',
        '"golang.org/x/sys/unix"\n\t"unsafe"')

with open(path, "w") as f:
    f.write(code)
print("Added RM replay to afterLoadImpl")
