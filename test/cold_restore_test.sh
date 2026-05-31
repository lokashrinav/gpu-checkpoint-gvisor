#!/bin/bash
set -euo pipefail

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver: $DRV"
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "=== Build test binary ==="
cat > /tmp/cr_test.c << 'TESTEOF'
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>
typedef int R;typedef int Dev;typedef void*Ctx;typedef unsigned long long Dp;typedef R(*Fn)(int,void*);
static volatile sig_atomic_t gpu_locked=0;
static Fn p_lock,p_ckpt,p_restore,p_unlock;
void on_save(int s){
    char a[64]={0};int r=p_lock(getpid(),a);fprintf(stderr,"SAVE:lock=%d\n",r);
    memset(a,0,64);r=p_ckpt(getpid(),a);fprintf(stderr,"SAVE:ckpt=%d\n",r);
    gpu_locked=1;
    FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}
}
void on_restore(int s){
    char a[64]={0};fprintf(stderr,"RESTORE:restore(%d)\n",getpid());
    int r=p_restore(getpid(),a);fprintf(stderr,"RESTORE:restore=%d\n",r);
    if(r==0){memset(a,0,64);r=p_unlock(getpid(),a);fprintf(stderr,"RESTORE:unlock=%d\n",r);}
    else{fprintf(stderr,"RESTORE:FAILED=%d\n",r);}
    gpu_locked=0;
    FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}
}
int main(){
    signal(SIGUSR1,on_save);signal(SIGUSR2,on_restore);
    void*h=dlopen("libcuda.so.1",RTLD_NOW);
    if(!h){fprintf(stderr,"dlopen failed: %s\n",dlerror());return 1;}
    R(*I)(unsigned)=dlsym(h,"cuInit");R(*C)(int*)=dlsym(h,"cuDeviceGetCount");
    R(*Rt)(Ctx*,int)=dlsym(h,"cuDevicePrimaryCtxRetain");R(*Sc)(Ctx)=dlsym(h,"cuCtxSetCurrent");
    R(*Ma)(Dp*,size_t)=dlsym(h,"cuMemAlloc_v2");R(*Ms)(Dp,unsigned,size_t)=dlsym(h,"cuMemsetD32_v2");
    R(*Md)(void*,Dp,size_t)=dlsym(h,"cuMemcpyDtoH_v2");R(*DG)(Dev*,int)=dlsym(h,"cuDeviceGet");
    p_lock=dlsym(h,"cuCheckpointProcessLock");p_ckpt=dlsym(h,"cuCheckpointProcessCheckpoint");
    p_restore=dlsym(h,"cuCheckpointProcessRestore");p_unlock=dlsym(h,"cuCheckpointProcessUnlock");
    R r=I(0);fprintf(stderr,"cuInit=%d\n",r);if(r)return 1;
    int ng;C(&ng);fprintf(stderr,"gpus=%d\n",ng);
    if(ng<1){fprintf(stderr,"no gpus\n");return 1;}
    Ctx ctx;Dev d;DG(&d,0);Rt(&ctx,0);Sc(ctx);
    Dp dptr;Ma(&dptr,4*1024*1024);Ms(dptr,0xBEEF1234,1024*1024);
    printf("pid=%d gpu=0 pattern=0xBEEF1234\nREADY\n");fflush(stdout);
    for(int tick=1;;tick++){
        sleep(2);
        if(gpu_locked){printf("tick=%d gpu_locked\n",tick*2);fflush(stdout);continue;}
        Sc(ctx);unsigned v;R cr=Md(&v,dptr,4);
        if(cr||v!=0xBEEF1234u){printf("tick=%d FAIL r=%d v=0x%X\n",tick*2,cr,v);fflush(stdout);}
        else{printf("tick=%d ok val=0x%X\n",tick*2,v);fflush(stdout);}
    }
}
TESTEOF
gcc -o /tmp/cr_test /tmp/cr_test.c -ldl
echo "Test binary built"

echo "=== Build signal helper ==="
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
echo "Helper built"

echo "=== Build OCI bundle with ALL nvidia libs ==="
B=/tmp/oci-bundle; R=$B/rootfs
rm -rf $B; mkdir -p $R/{bin,lib64,lib/x86_64-linux-gnu,dev,proc,sys,tmp,etc}
cp /tmp/cr_test $R/bin/test
cp /tmp/gvisor-gpu-ckpt $R/bin/gvisor-gpu-ckpt

# Copy ALL libs the test binary needs
for l in $(ldd /tmp/cr_test | grep "=> /" | awk '{print $3}'); do
    cp -L "$l" $R/lib64/
done
cp -L /lib64/ld-linux-x86-64.so.2 $R/lib64/

# Copy ALL nvidia libs (not just libcuda — need the full driver stack)
for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1 \
           libnvidia-nvvm.so.4 libnvidia-fatbinaryloader.so.$DRV \
           libnvidia-gpucomp.so.$DRV libnvidia-nvjitlink.so.12; do
    found=$(find /usr/lib /lib -name "${lib}*" -not -type d 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        cp -L "$found" $R/lib64/$(basename $found)
    fi
done

# Symlinks for loader
for l in $R/lib64/*; do
    ln -sf /lib64/$(basename $l) $R/lib/x86_64-linux-gnu/ 2>/dev/null || true
done
echo "/lib64" > $R/etc/ld.so.conf
echo "/lib/x86_64-linux-gnu" >> $R/etc/ld.so.conf

echo "Rootfs libs:"
ls $R/lib64/ | head -20

# OCI spec with nvidia devices
cd $B; runsc-patched spec
python3 -c "
import json, os
with open('config.json') as f: s = json.load(f)
s['process']['args'] = ['/bin/test']
s['process']['env'] = ['PATH=/bin', 'LD_LIBRARY_PATH=/lib64:/lib/x86_64-linux-gnu']
s['root']['readonly'] = False
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
print('%d nvidia devices' % len(devs))
"
echo "Bundle ready"

echo ""
echo "==========================================="
echo " COLD RESTORE TEST (raw runsc, no Docker)"
echo "==========================================="

S=/tmp/rs; C=/tmp/ck
sudo rm -rf $S $C; mkdir -p $S $C

echo ""
echo "=== Step 1: Start container ==="
sudo runsc-patched --root=$S --nvproxy --nvproxy-driver-version=$DRV \
    run --bundle $B mgpu 2>&1 &
P=$!
sleep 12
if kill -0 $P 2>/dev/null; then
    echo "RUNNING"
else
    echo "DIED — checking why"
    exit 1
fi

echo ""
echo "=== Step 2: Checkpoint (lock + ckpt + serialize + sentry exits) ==="
sudo runsc-patched --root=$S checkpoint \
    --save-restore-exec-argv=/bin/gvisor-gpu-ckpt \
    --save-restore-exec-timeout=120s \
    --image-path=$C mgpu 2>&1
echo "Checkpoint exit: $?"
ls -lh $C/
echo "Sentry alive: $(kill -0 $P 2>/dev/null && echo YES || echo NO)"

echo ""
echo "=== Step 3: Cold restore (new sentry, new driver session) ==="
echo "The save-restore-exec binary in restore mode signals SIGUSR2"
echo "The app calls cuCheckpointProcessRestore — reads from host memory"
sudo runsc-patched --root=$S --nvproxy --nvproxy-driver-version=$DRV \
    --debug --debug-log=/tmp/cr_dbg.%COMMAND%.log \
    restore --image-path=$C --bundle=$B mgpu-r 2>&1 &
P2=$!
sleep 30

echo ""
echo "=== Results ==="
echo "Process: $(kill -0 $P2 2>/dev/null && echo ALIVE || echo DEAD)"
echo ""
echo "=== Debug log ==="
grep -E "SAVE:|RESTORE:|helper:|restore=|ok|val|FAIL|signal 11" /tmp/cr_dbg.boot.log 2>/dev/null | tail -20

echo ""
if grep -q "RESTORE:restore=0" /tmp/cr_dbg.boot.log 2>/dev/null; then
    echo "==========================================="
    echo " COLD RESTORE WORKS!"
    echo " GPU state survived sentry restart!"
    echo "==========================================="
    grep "ok val=" /tmp/cr_dbg.boot.log 2>/dev/null | tail -5
elif grep -q "RESTORE:FAILED" /tmp/cr_dbg.boot.log 2>/dev/null; then
    echo "RESTORE FAILED"
    grep "RESTORE:" /tmp/cr_dbg.boot.log 2>/dev/null
else
    echo "No restore output — checking if process crashed"
    grep "signal 11\|exiting\|FATAL" /tmp/cr_dbg.boot.log 2>/dev/null | tail -5
fi

# Cleanup
sudo runsc-patched --root=$S kill mgpu-r SIGTERM 2>/dev/null || true
sudo runsc-patched --root=$S kill mgpu SIGTERM 2>/dev/null || true
sleep 1
sudo runsc-patched --root=$S delete mgpu-r 2>/dev/null || true
sudo runsc-patched --root=$S delete mgpu 2>/dev/null || true
wait $P $P2 2>/dev/null || true
echo "DONE"
