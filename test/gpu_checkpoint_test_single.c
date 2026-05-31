/* Single-GPU checkpoint test. Verified on H100, driver 580. */
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
void on_save(int s){char a[64]={0};int r=p_lock(getpid(),a);fprintf(stderr,"SAVE:lock=%d\n",r);memset(a,0,64);r=p_ckpt(getpid(),a);fprintf(stderr,"SAVE:ckpt=%d\n",r);gpu_locked=1;FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}}
void on_restore(int s){char a[64]={0};fprintf(stderr,"RESTORE:restore(%d)\n",getpid());int r=p_restore(getpid(),a);fprintf(stderr,"RESTORE:restore=%d\n",r);if(r==0){memset(a,0,64);r=p_unlock(getpid(),a);fprintf(stderr,"RESTORE:unlock=%d\n",r);}gpu_locked=0;FILE*f=fopen("/tmp/.gpu_ckpt_done","w");if(f){fprintf(f,"0\n");fclose(f);}}
int main(){
    signal(SIGUSR1,on_save);signal(SIGUSR2,on_restore);
    void*h=dlopen("libcuda.so.1",RTLD_NOW);
    R(*I)(unsigned)=dlsym(h,"cuInit");R(*Rt)(Ctx*,int)=dlsym(h,"cuDevicePrimaryCtxRetain");
    R(*Sc)(Ctx)=dlsym(h,"cuCtxSetCurrent");R(*Ma)(Dp*,size_t)=dlsym(h,"cuMemAlloc_v2");
    R(*Ms)(Dp,unsigned,size_t)=dlsym(h,"cuMemsetD32_v2");R(*Md)(void*,Dp,size_t)=dlsym(h,"cuMemcpyDtoH_v2");
    p_lock=dlsym(h,"cuCheckpointProcessLock");p_ckpt=dlsym(h,"cuCheckpointProcessCheckpoint");
    p_restore=dlsym(h,"cuCheckpointProcessRestore");p_unlock=dlsym(h,"cuCheckpointProcessUnlock");
    I(0);Ctx ctx;Dp dptr;Dev d;R(*DG)(Dev*,int)=dlsym(h,"cuDeviceGet");DG(&d,0);Rt(&ctx,0);Sc(ctx);
    Ma(&dptr,4*1024*1024);Ms(dptr,0xBEEF0001,1024*1024);
    printf("pid=%d single-GPU\nGPU 0: 0xBEEF0001\nREADY\n");fflush(stdout);
    for(int tick=1;;tick++){sleep(2);
        if(gpu_locked){printf("tick=%d gpu_locked\n",tick*2);fflush(stdout);continue;}
        Sc(ctx);unsigned v;R cr=Md(&v,dptr,4);
        if(cr||v!=0xBEEF0001u)printf("GPU0:r=%d v=0x%X FAIL\n",cr,v);
        else printf("tick=%d ok\n",tick*2);fflush(stdout);
    }
}
