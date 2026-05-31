"""
PyTorch GPU checkpoint/restore test.
Verified on 2x H100 SXM5, driver 580.105.08, PyTorch + gVisor.

Result: tensors_ok=True model_ok=True after full cuda-checkpoint cycle.
Checkpoint file: 945KB state + 408MB pages.
"""
import torch
import signal
import os
import sys
import ctypes
import time

libcuda = ctypes.CDLL("libcuda.so.1")

class CheckpointArgs(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * 64)]

def ckpt_call(fn, pid):
    args = CheckpointArgs()
    return fn(pid, ctypes.byref(args))

gpu_locked = False

def save_handler(signum, frame):
    global gpu_locked
    pid = os.getpid()
    r = ckpt_call(libcuda.cuCheckpointProcessLock, pid)
    print(f"SAVE:lock={r}", file=sys.stderr, flush=True)
    r = ckpt_call(libcuda.cuCheckpointProcessCheckpoint, pid)
    print(f"SAVE:ckpt={r}", file=sys.stderr, flush=True)
    gpu_locked = True
    with open("/tmp/.gpu_ckpt_done", "w") as f: f.write("0\n")

def restore_handler(signum, frame):
    global gpu_locked
    pid = os.getpid()
    r = ckpt_call(libcuda.cuCheckpointProcessRestore, pid)
    print(f"RESTORE:restore={r}", file=sys.stderr, flush=True)
    r = ckpt_call(libcuda.cuCheckpointProcessUnlock, pid)
    print(f"RESTORE:unlock={r}", file=sys.stderr, flush=True)
    gpu_locked = False
    with open("/tmp/.gpu_ckpt_done", "w") as f: f.write("0\n")

signal.signal(signal.SIGUSR1, save_handler)
signal.signal(signal.SIGUSR2, restore_handler)

print(f"pid={os.getpid()}")
n_gpus = torch.cuda.device_count()
print(f"PyTorch sees {n_gpus} GPUs")

tensors = []
for i in range(min(n_gpus, 2)):
    t = torch.full((1024, 1024), 0xCAFE + i, dtype=torch.int32, device=f"cuda:{i}")
    tensors.append(t)
    print(f"GPU {i}: tensor shape={t.shape} val={t[0,0].item():#x}")

model = torch.nn.Linear(1024, 1024).cuda(0)
x = torch.randn(32, 1024).cuda(0)
y = model(x)
print(f"Model output shape: {y.shape}, sum={y.sum().item():.2f}")
print("READY", flush=True)

tick = 0
while True:
    time.sleep(2)
    tick += 1
    if gpu_locked:
        print(f"tick={tick*2} gpu_locked", flush=True)
        continue
    ok = True
    for i, t in enumerate(tensors):
        expected = 0xCAFE + i
        if t[0, 0].item() != expected:
            ok = False
    try:
        y2 = model(x)
        model_ok = y2.shape == y.shape
    except:
        model_ok = False
        ok = False
    print(f"tick={tick*2} tensors_ok={ok} model_ok={model_ok}", flush=True)
