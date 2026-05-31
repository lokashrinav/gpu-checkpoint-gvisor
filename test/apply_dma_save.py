#!/usr/bin/env python3
"""Apply DMA GPU save to InvalidateUnsavable in save_restore.go"""

path = "/tmp/gvisor/pkg/sentry/mm/save_restore.go"
with open(path) as f:
    code = f.read()

old = """\tmm.activeMu.Lock()
\tdefer mm.activeMu.Unlock()
\tpseg := mm.pmas.FirstSegment()
\tfor pseg.Ok() {
\t\tif _, ok := pseg.ValuePtr().file.(*pgalloc.MemoryFile); !ok {
\t\t\tmm.unmapASLocked(pseg.Range())
\t\t\tmm.removeRSSLocked(pseg.Range())
\t\t\tpseg.ValuePtr().file.DecRef(pseg.fileRange())
\t\t\tpseg = mm.pmas.Remove(pseg).NextSegment()
\t\t} else {
\t\t\tpseg = pseg.NextSegment()
\t\t}
\t}"""

new = """\tmm.activeMu.Lock()
\tdefer mm.activeMu.Unlock()
\tmm.savedGPUPages = make(map[uint64][]byte)
\tpseg := mm.pmas.FirstSegment()
\tfor pseg.Ok() {
\t\tif _, ok := pseg.ValuePtr().file.(*pgalloc.MemoryFile); !ok {
\t\t\tar := pseg.Range()
\t\t\tfr := pseg.fileRange()
\t\t\tbs, err := pseg.ValuePtr().file.MapInternal(fr, hostarch.Read)
\t\t\tif err == nil && bs.NumBytes() > 0 {
\t\t\t\tbuf := make([]byte, bs.NumBytes())
\t\t\t\tsafemem.CopySeq(
\t\t\t\t\tsafemem.BlockSeqOf(safemem.BlockFromSafeSlice(buf)),
\t\t\t\t\tbs,
\t\t\t\t)
\t\t\t\tmm.savedGPUPages[uint64(ar.Start)] = buf
\t\t\t\tlog.Infof("GPU checkpoint: saved %d bytes at vaddr %#x", len(buf), ar.Start)
\t\t\t}
\t\t\tmm.unmapASLocked(pseg.Range())
\t\t\tmm.removeRSSLocked(pseg.Range())
\t\t\tpseg.ValuePtr().file.DecRef(pseg.fileRange())
\t\t\tpseg = mm.pmas.Remove(pseg).NextSegment()
\t\t} else {
\t\t\tpseg = pseg.NextSegment()
\t\t}
\t}"""

assert old in code, "Pattern not found in save_restore.go"
code = code.replace(old, new)

# Add imports
safemem_import = '\t"gvisor.dev/gvisor/pkg/safemem"'
log_import = '\t"gvisor.dev/gvisor/pkg/log"'
hostarch_line = '\t"gvisor.dev/gvisor/pkg/hostarch"'

if safemem_import not in code:
    code = code.replace(hostarch_line, hostarch_line + "\n" + log_import + "\n" + safemem_import)
elif log_import not in code:
    code = code.replace(hostarch_line, hostarch_line + "\n" + log_import)

with open(path, "w") as f:
    f.write(code)
print("save_restore.go: OK")
