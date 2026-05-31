#!/bin/bash
# Multi-GPU cold restore + NCCL test
# Run on a machine with 2+ Turing+ GPUs, gVisor installed, persistence mode on
set -euo pipefail

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
NUM=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Driver: $DRV, GPUs: $NUM"

if [ "$NUM" -lt 2 ]; then echo "NEED 2+ GPUS"; exit 1; fi

echo "=== Setup ==="
sudo apt-get update -qq
sudo apt-get install -y -qq golang-go clang llvm libbpf-dev gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-i386 libnccl2 libnccl-dev > /dev/null 2>&1
which bazel > /dev/null 2>&1 || { wget -q https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O /tmp/bz; chmod +x /tmp/bz; sudo cp /tmp/bz /usr/local/bin/bazel; }

if ! which runsc-patched > /dev/null 2>&1; then
    cd /tmp; rm -rf gvisor gpu-checkpoint-gvisor
    git clone --quiet https://github.com/lokashrinav/gpu-checkpoint-gvisor.git
    git clone --quiet --depth=1 https://github.com/google/gvisor.git
    cd gvisor; sudo chown -R ubuntu:ubuntu .
    git apply /tmp/gpu-checkpoint-gvisor/gvisor-nvproxy-checkpoint.patch
    bazel build //runsc:runsc --jobs=40 2>&1 | tail -3
    RUNSC=$(find ~/.cache/bazel -name runsc -path "*/runsc_/*" -type f 2>/dev/null | head -1)
    sudo cp "$RUNSC" /usr/local/bin/runsc-patched
fi
echo "runsc: $(runsc-patched --version 2>&1 | head -1)"
sudo nvidia-smi -pm 1 2>&1 | tail -1

echo "=== Build test binary ==="
cat > /tmp/nccl_cold_test.cu << 'TESTEOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(c) do{cudaError_t e=c;if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CHECK_NCCL(c) do{ncclResult_t e=c;if(e){fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(e));exit(1);}}while(0)

typedef int R; typedef R(*Fn)(int,void*);
static volatile sig_atomic_t gpu_locked=0;
static Fn p_lock,p_ckpt,p_restore,p_unlock;
static int ng;
static ncclComm_t comms[8];

void on_save(int s){
    // Suspend NCCL first
    for(int i=0;i<ng;i++){
        ncclResult_t nr=ncclCommSuspend(comms[i]);
        fprintf(stderr,"SAVE:ncclSuspend[%d]=%d\n",i,nr);
    }
    // Then lock + checkpoint
    char a[64]={0};int r=p_lock(getpid(),a);fprintf(stderr,"SAVE:lock=%d\n",r);
    memset(a,0,64);r=p_ckpt(getpid(),a);fprintf(stderr,"SAVE:ckpt=%d\n",r);
    gpu_locked=1;
    FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}
}
void on_restore(int s){
    // Restore + unlock
    char a[64]={0};fprintf(stderr,"RESTORE:restore(%d)\n",getpid());
    int r=p_restore(getpid(),a);fprintf(stderr,"RESTORE:restore=%d\n",r);
    if(r==0){memset(a,0,64);r=p_unlock(getpid(),a);fprintf(stderr,"RESTORE:unlock=%d\n",r);}
    else{fprintf(stderr,"RESTORE:FAILED=%d\n",r);}
    // Resume NCCL
    for(int i=0;i<ng;i++){
        ncclResult_t nr=ncclCommResume(comms[i]);
        fprintf(stderr,"RESTORE:ncclResume[%d]=%d\n",i,nr);
    }
    gpu_locked=0;
    FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}
}

int main(){
    signal(SIGUSR1,on_save);signal(SIGUSR2,on_restore);
    void*h=dlopen("libcuda.so.1",RTLD_NOW);
    p_lock=dlsym(h,"cuCheckpointProcessLock");p_ckpt=dlsym(h,"cuCheckpointProcessCheckpoint");
    p_restore=dlsym(h,"cuCheckpointProcessRestore");p_unlock=dlsym(h,"cuCheckpointProcessUnlock");

    CHECK_CUDA(cudaGetDeviceCount(&ng));
    if(ng>2)ng=2;
    printf("pid=%d gpus=%d\n",getpid(),ng);
    if(ng<2){fprintf(stderr,"Need 2+ GPUs\n");return 1;}

    // Allocate + pattern
    size_t count=1024*1024;
    float *d_s[2],*d_r[2];
    cudaStream_t streams[2];
    for(int i=0;i<ng;i++){
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_s[i],count*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_r[i],count*sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        float val=(float)(i+1);
        float *hbuf=malloc(count*sizeof(float));
        for(size_t j=0;j<count;j++)hbuf[j]=val;
        CHECK_CUDA(cudaMemcpy(d_s[i],hbuf,count*sizeof(float),cudaMemcpyHostToDevice));
        free(hbuf);
        printf("GPU %d: filled %.1f\n",i,val);
    }

    // NCCL init + allreduce
    int devs[2]={0,1};
    CHECK_NCCL(ncclCommInitAll(comms,ng,devs));
    printf("NCCL initialized\n");
    CHECK_NCCL(ncclGroupStart());
    for(int i=0;i<ng;i++)CHECK_NCCL(ncclAllReduce(d_s[i],d_r[i],count,ncclFloat,ncclSum,comms[i],streams[i]));
    CHECK_NCCL(ncclGroupEnd());
    for(int i=0;i<ng;i++){CHECK_CUDA(cudaSetDevice(i));CHECK_CUDA(cudaStreamSynchronize(streams[i]));}
    float expected=0;for(int i=0;i<ng;i++)expected+=(float)(i+1);
    printf("Allreduce done, expected=%.1f\n",expected);
    for(int i=0;i<ng;i++){
        CHECK_CUDA(cudaSetDevice(i));float v;
        CHECK_CUDA(cudaMemcpy(&v,d_r[i],sizeof(float),cudaMemcpyDeviceToHost));
        printf("GPU %d: allreduce=%.1f %s\n",i,v,v==expected?"OK":"MISMATCH");
    }

    // Write verification patterns
    for(int i=0;i<ng;i++){
        CHECK_CUDA(cudaSetDevice(i));
        unsigned int pat=0xCAFE0000|i;
        unsigned int *hbuf=malloc(count*sizeof(unsigned int));
        for(size_t j=0;j<count;j++)hbuf[j]=pat;
        CHECK_CUDA(cudaMemcpy(d_r[i],hbuf,count*sizeof(unsigned int),cudaMemcpyHostToDevice));
        free(hbuf);
        printf("GPU %d: pattern 0x%X\n",i,pat);
    }
    printf("READY\n");fflush(stdout);

    for(int tick=1;;tick++){
        sleep(2);
        if(gpu_locked){printf("tick=%d gpu_locked\n",tick*2);fflush(stdout);continue;}

        // Verify patterns
        int ok=1;
        for(int i=0;i<ng;i++){
            CHECK_CUDA(cudaSetDevice(i));
            unsigned int exp=0xCAFE0000|i,val;
            CHECK_CUDA(cudaMemcpy(&val,d_r[i],sizeof(unsigned int),cudaMemcpyDeviceToHost));
            if(val!=exp){printf("GPU%d:0x%X!=0x%X\n",i,val,exp);ok=0;}
        }

        // Try NCCL allreduce to verify communication still works
        int nccl_ok=1;
        CHECK_NCCL(ncclGroupStart());
        for(int i=0;i<ng;i++)CHECK_NCCL(ncclAllReduce(d_s[i],d_r[i],count,ncclFloat,ncclSum,comms[i],streams[i]));
        ncclResult_t nr=ncclGroupEnd();
        if(nr!=ncclSuccess){nccl_ok=0;}
        else{
            for(int i=0;i<ng;i++){CHECK_CUDA(cudaSetDevice(i));CHECK_CUDA(cudaStreamSynchronize(streams[i]));}
            // Restore patterns after allreduce overwrote d_r
            for(int i=0;i<ng;i++){
                CHECK_CUDA(cudaSetDevice(i));
                unsigned int pat=0xCAFE0000|i;
                unsigned int *hbuf=malloc(count*sizeof(unsigned int));
                for(size_t j=0;j<count;j++)hbuf[j]=pat;
                CHECK_CUDA(cudaMemcpy(d_r[i],hbuf,count*sizeof(unsigned int),cudaMemcpyHostToDevice));
                free(hbuf);
            }
        }

        printf("tick=%d patterns_ok=%d nccl_ok=%d\n",tick*2,ok,nccl_ok);fflush(stdout);
    }
}
TESTEOF
nvcc -Wno-deprecated-gpu-targets -o /tmp/nccl_cold_test /tmp/nccl_cold_test.cu -lnccl -lcuda -ldl
echo "Test binary built"

echo "=== Build helper ==="
if [ ! -f /tmp/gvisor-gpu-ckpt ]; then
    cat > /tmp/helper.go << 'GOEOF'
package main
import ("fmt";"os";"strings";"syscall";"time")
func main() {
    mode := os.Getenv("GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE")
    fmt.Fprintf(os.Stderr, "helper: mode=%s\n", mode)
    var sig syscall.Signal
    switch mode {
    case "save": sig = syscall.SIGUSR1
    case "restore","resume": sig = syscall.SIGUSR2
    default: return
    }
    os.Remove("/tmp/.gpu_ckpt_done")
    syscall.Kill(1, sig)
    for i := 0; i < 120; i++ {
        if d, err := os.ReadFile("/tmp/.gpu_ckpt_done"); err == nil {
            rc := strings.TrimSpace(string(d))
            fmt.Fprintf(os.Stderr, "helper: %s rc=%s\n", mode, rc)
            if rc != "0" { os.Exit(1) }
            return
        }
        time.Sleep(500*time.Millisecond)
    }
    os.Exit(1)
}
GOEOF
    mkdir -p /tmp/hb; cp /tmp/helper.go /tmp/hb/main.go
    cd /tmp/hb; go mod init h 2>/dev/null || true; go build -o /tmp/gvisor-gpu-ckpt .
fi
echo "Helper ready"

echo "=== Build OCI bundle ==="
B=/tmp/oci-bundle; R=$B/rootfs
rm -rf $B; mkdir -p $R/{bin,lib64,lib/x86_64-linux-gnu,dev,proc,sys,tmp,etc}
cp /tmp/nccl_cold_test $R/bin/test
cp /tmp/gvisor-gpu-ckpt $R/bin/gvisor-gpu-ckpt

# ALL libs
for l in $(ldd /tmp/nccl_cold_test | grep "=> /" | awk '{print $3}'); do cp -L "$l" $R/lib64/; done
cp -L /lib64/ld-linux-x86-64.so.2 $R/lib64/
for lib in $(find /usr/lib/x86_64-linux-gnu -name "libnvidia-*.so*" -not -type d 2>/dev/null); do
    cp -L "$lib" $R/lib64/$(basename $lib) 2>/dev/null || true
done
for lib in libcuda.so.1 libnccl.so.2 libnvidia-ml.so.1; do
    found=$(find /usr/lib /lib -name "${lib}*" -not -type d 2>/dev/null | head -1)
    [ -n "$found" ] && cp -L "$found" $R/lib64/$lib
done
for lib in libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 libgcc_s.so.1 libstdc++.so.6; do
    found=$(find /lib /usr/lib -name "$lib" 2>/dev/null | head -1)
    [ -n "$found" ] && cp -L "$found" $R/lib64/
done
for l in $R/lib64/*; do ln -sf /lib64/$(basename $l) $R/lib/x86_64-linux-gnu/ 2>/dev/null || true; done
echo "/lib64" > $R/etc/ld.so.conf

cd $B; runsc-patched spec
python3 -c "
import json, os
with open('config.json') as f: s = json.load(f)
s['process']['args'] = ['/bin/test']
s['process']['env'] = ['PATH=/bin', 'LD_LIBRARY_PATH=/lib64:/lib/x86_64-linux-gnu', 'NCCL_SOCKET_IFNAME=lo', 'NCCL_DEBUG=WARN']
s['root']['readonly'] = False
ns = s.get('linux',{}).get('namespaces',[])
s['linux']['namespaces'] = [n for n in ns if n.get('type') != 'network']
devs = []
for i in range(8):
    d = '/dev/nvidia%d' % i
    if os.path.exists(d):
        st = os.stat(d)
        devs.append({'path':d,'type':'c','major':os.major(st.st_rdev),'minor':os.minor(st.st_rdev),'fileMode':438})
for n in ['nvidiactl','nvidia-uvm','nvidia-uvm-tools']:
    d = '/dev/' + n
    if os.path.exists(d):
        st = os.stat(d)
        devs.append({'path':d,'type':'c','major':os.major(st.st_rdev),'minor':os.minor(st.st_rdev),'fileMode':438})
s['linux']['devices'] = devs
s['linux']['resources'] = {'devices':[{'allow':True,'access':'rwm'}]}
with open('config.json','w') as f: json.dump(s,f,indent=2)
print('%d devices' % len(devs))
"
echo "Bundle ready"

echo ""
echo "==========================================="
echo " MULTI-GPU COLD RESTORE + NCCL"
echo " ${NUM} GPUs | raw runsc | sentry dies"
echo "==========================================="

S=/tmp/rs; C=/tmp/ck
sudo rm -rf $S $C; mkdir -p $S $C

echo ""
echo "=== Step 1: Start (NCCL allreduce + patterns) ==="
sudo runsc-patched --root=$S --nvproxy --nvproxy-driver-version=$DRV \
    --network=host run --bundle $B mgpu 2>&1 &
P=$!; sleep 15
if ! kill -0 $P 2>/dev/null; then
    echo "Host networking failed, trying sandbox networking"
    sudo rm -rf $S; mkdir -p $S
    python3 -c "
import json
with open('$B/config.json') as f: s = json.load(f)
ns = s.get('linux',{}).get('namespaces',[])
if not any(n.get('type')=='network' for n in ns): ns.append({'type':'network'})
s['linux']['namespaces'] = ns
with open('$B/config.json','w') as f: json.dump(s,f,indent=2)
"
    sudo runsc-patched --root=$S --nvproxy --nvproxy-driver-version=$DRV \
        run --bundle $B mgpu 2>&1 &
    P=$!; sleep 15
fi
kill -0 $P && echo "RUNNING" || { echo "DIED"; exit 1; }

echo ""
echo "=== Step 2: Checkpoint (ncclSuspend + lock + ckpt + sentry exits) ==="
sudo runsc-patched --root=$S checkpoint \
    --save-restore-exec-argv=/bin/gvisor-gpu-ckpt \
    --save-restore-exec-timeout=120s \
    --image-path=$C mgpu 2>&1
echo "Checkpoint: $?"
ls -lh $C/

echo ""
echo "=== Step 3: Cold restore (new sentry + restore + ncclResume) ==="
sudo runsc-patched --root=$S --nvproxy --nvproxy-driver-version=$DRV \
    --debug --debug-log=/tmp/mcr.%COMMAND%.log \
    restore --image-path=$C --bundle=$B mgpu-r 2>&1 &
P2=$!; sleep 30

echo ""
echo "=== RESULTS ==="
echo "Process: $(kill -0 $P2 2>/dev/null && echo ALIVE || echo DEAD)"
echo ""
grep -E "SAVE:|RESTORE:|ncclSuspend|ncclResume|patterns_ok|nccl_ok|FAIL|signal 11" /tmp/mcr.boot.log 2>/dev/null | tail -25

echo ""
if grep -q "patterns_ok=1 nccl_ok=1" /tmp/mcr.boot.log 2>/dev/null; then
    echo "==========================================="
    echo " SUCCESS: MULTI-GPU COLD RESTORE + NCCL!"
    echo " Patterns verified + NCCL communication OK"
    echo "==========================================="
    grep "patterns_ok=1 nccl_ok=1" /tmp/mcr.boot.log 2>/dev/null | head -5
elif grep -q "RESTORE:restore=0" /tmp/mcr.boot.log 2>/dev/null; then
    echo "RESTORE=0 but checking NCCL..."
    grep "nccl_ok" /tmp/mcr.boot.log 2>/dev/null | tail -5
else
    echo "FAILED"
    grep -E "signal 11|FAILED|error" /tmp/mcr.boot.log 2>/dev/null | tail -5
fi

sudo runsc-patched --root=$S kill mgpu-r SIGTERM 2>/dev/null || true
sudo runsc-patched --root=$S kill mgpu SIGTERM 2>/dev/null || true
sleep 1
sudo runsc-patched --root=$S delete mgpu-r 2>/dev/null || true
sudo runsc-patched --root=$S delete mgpu 2>/dev/null || true
wait $P $P2 2>/dev/null || true
echo "DONE"
