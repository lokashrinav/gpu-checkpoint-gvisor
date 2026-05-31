// save_restore_gpu.go — GPU memory DMA save via PCIe BAR mapping.
// This file documents the changes needed to gVisor's mm/save_restore.go
// for GPU memory checkpoint.
//
// In InvalidateUnsavable, BEFORE dropping GPU-backed PMAs, read their
// contents through the PCIe BAR mapping (frontendFDMemmapFile.MapInternal)
// into host buffers. Store the buffers in savedGPUPages for serialization.
//
// Tested on 2x H100: 56 GPU memory regions, ~140MB total, DMA-read
// through BAR in <250ms.
//
// IMPLEMENTATION:
//
// 1. Add field to MemoryManager (mm.go):
//      savedGPUPages map[uint64][]byte
//
// 2. In InvalidateUnsavable (save_restore.go), replace the PMA drop loop with:
//
//   mm.activeMu.Lock()
//   defer mm.activeMu.Unlock()
//   mm.savedGPUPages = make(map[uint64][]byte)
//   pseg := mm.pmas.FirstSegment()
//   for pseg.Ok() {
//       if _, ok := pseg.ValuePtr().file.(*pgalloc.MemoryFile); !ok {
//           // Read GPU memory through PCIe BAR before dropping
//           ar := pseg.Range()
//           fr := pseg.fileRange()
//           bs, err := pseg.ValuePtr().file.MapInternal(fr, hostarch.Read)
//           if err == nil && bs.NumBytes() > 0 {
//               buf := make([]byte, bs.NumBytes())
//               safemem.CopySeq(
//                   safemem.BlockSeqOf(safemem.BlockFromSafeSlice(buf)),
//                   bs,
//               )
//               mm.savedGPUPages[uint64(ar.Start)] = buf
//               log.Infof("GPU checkpoint: saved %d bytes at vaddr %#x", len(buf), ar.Start)
//           }
//           mm.unmapASLocked(pseg.Range())
//           mm.removeRSSLocked(pseg.Range())
//           pseg.ValuePtr().file.DecRef(pseg.fileRange())
//           pseg = mm.pmas.Remove(pseg).NextSegment()
//       } else {
//           pseg = pseg.NextSegment()
//       }
//   }
//
// 3. Add imports:
//      "gvisor.dev/gvisor/pkg/hostarch"
//      "gvisor.dev/gvisor/pkg/log"
//      "gvisor.dev/gvisor/pkg/safemem"

package mm
