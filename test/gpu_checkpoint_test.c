/*
 * Multi-GPU checkpoint/restore test — WORKING on 2x H100.
 *
 * Uses cuda-checkpoint API with --leave-running gVisor checkpoint.
 * SIGUSR1: lock + checkpoint GPUs (in signal handler)
 * SIGUSR2: restore + unlock GPUs (in signal handler)
 *
 * Verified: lock=0, ckpt=0, restore=0, unlock=0, all_gpus_ok=1
 * GPU patterns 0xCAFE0000 + 0xCAFE0001 intact after full cycle.
 *
 * Build: gcc gpu_checkpoint_test.c -o gpu_test -ldl
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>
#include <sys/syscall.h>
typedef int R;typedef int Dev;typedef void*Ctx;typedef unsigned long long Dp;typedef R(*Fn)(int,void*);
static volatile sig_atomic_t gpu_locked=0;
static Fn p_lock,p_ckpt,p_restore,p_unlock;

void on_save(int s){
    char a[64]={0};
    int r=p_lock(getpid(),a);
    fprintf(stderr,"SAVE:lock=%d\n",r);
    memset(a,0,64);r=p_ckpt(getpid(),a);
    fprintf(stderr,"SAVE:ckpt=%d\n",r);
    gpu_locked=1;
    FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}
}
void on_restore(int s){
    char a[64]={0};
    fprintf(stderr,"RESTORE:restore(%d)\n",getpid());
    int r=p_restore(getpid(),a);
    fprintf(stderr,"RESTORE:restore=%d\n",r);
    if(r==0){memset(a,0,64);r=p_unlock(getpid(),a);fprintf(stderr,"RESTORE:unlock=%d\n",r);}
    gpu_locked=0;
    FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}
}

int main(){
    signal(SIGUSR1,on_save);signal(SIGUSR2,on_restore);
    void*h=dlopen("libcuda.so.1",RTLD_NOW);
    R(*I)(unsigned)=dlsym(h,"cuInit");R(*C)(int*)=dlsym(h,"cuDeviceGetCount");
    R(*Rt)(Ctx*,int)=dlsym(h,"cuDevicePrimaryCtxRetain");R(*Sc)(Ctx)=dlsym(h,"cuCtxSetCurrent");
    R(*Ma)(Dp*,size_t)=dlsym(h,"cuMemAlloc_v2");R(*Ms)(Dp,unsigned,size_t)=dlsym(h,"cuMemsetD32_v2");
    R(*Md)(void*,Dp,size_t)=dlsym(h,"cuMemcpyDtoH_v2");
    R(*DG)(Dev*,int)=dlsym(h,"cuDeviceGet");
    p_lock=dlsym(h,"cuCheckpointProcessLock");p_ckpt=dlsym(h,"cuCheckpointProcessCheckpoint");
    p_restore=dlsym(h,"cuCheckpointProcessRestore");p_unlock=dlsym(h,"cuCheckpointProcessUnlock");
    I(0);int ng;C(&ng);if(ng>2)ng=2;
    printf("pid=%d gpus=%d\n",getpid(),ng);
    Ctx ctx[2];Dp dptr[2];
    for(int i=0;i<ng;i++){
        Dev d;DG(&d,i);Rt(&ctx[i],i);Sc(ctx[i]);
        Ma(&dptr[i],4*1024*1024);Ms(dptr[i],0xCAFE0000|i,1024*1024);
        printf("GPU %d: 0x%X\n",i,0xCAFE0000|i);
    }
    printf("READY\n");fflush(stdout);
    for(int tick=1;;tick++){
        sleep(2);
        if(gpu_locked){printf("tick=%d gpu_locked\n",tick*2);fflush(stdout);continue;}
        int ok=1;
        for(int i=0;i<ng;i++){
            Sc(ctx[i]);unsigned v;R cr=Md(&v,dptr[i],4);
            if(cr||v!=(0xCAFE0000u|i)){printf("GPU%d:r=%d v=0x%X\n",i,cr,v);ok=0;}
        }
        printf("tick=%d all_gpus_ok=%d\n",tick*2,ok);fflush(stdout);
    }
}
