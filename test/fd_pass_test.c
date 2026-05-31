/*
 * FD-passing experiment: does keeping the nvidia struct file alive
 * via SCM_RIGHTS allow cross-process cuCheckpointProcessRestore?
 *
 * Three processes:
 *   ./fd_pass_test save     - checkpoint GPU, pass FDs to keeper, exit
 *   ./fd_pass_test keep     - hold FDs alive (run in background)
 *   ./fd_pass_test restore  - receive FDs from keeper, call restore
 *
 * Build: gcc -o fd_pass_test fd_pass_test.c -ldl -lnvidia-ml -lpthread
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>
#include <nvml.h>

#define SOCK_PATH "/tmp/.gpu_fd_keeper.sock"
#define MAX_FDS 16

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef CUresult (*fn_t)(int, void*);

static void *cuda_handle;
static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGetCount)(int*);
static CUresult (*p_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuMemAlloc)(CUdeviceptr*, size_t);
static CUresult (*p_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
static CUresult (*p_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
static fn_t p_lock, p_ckpt, p_restore, p_unlock;
static CUresult (*p_getRestoreTid)(int, int*);

static void load_cuda() {
    cuda_handle = dlopen("libcuda.so.1", RTLD_NOW);
    p_cuInit = dlsym(cuda_handle, "cuInit");
    p_cuDeviceGetCount = dlsym(cuda_handle, "cuDeviceGetCount");
    p_cuDevicePrimaryCtxRetain = dlsym(cuda_handle, "cuDevicePrimaryCtxRetain");
    p_cuCtxSetCurrent = dlsym(cuda_handle, "cuCtxSetCurrent");
    p_cuMemAlloc = dlsym(cuda_handle, "cuMemAlloc_v2");
    p_cuMemsetD32 = dlsym(cuda_handle, "cuMemsetD32_v2");
    p_cuMemcpyDtoH = dlsym(cuda_handle, "cuMemcpyDtoH_v2");
    p_lock = dlsym(cuda_handle, "cuCheckpointProcessLock");
    p_ckpt = dlsym(cuda_handle, "cuCheckpointProcessCheckpoint");
    p_restore = dlsym(cuda_handle, "cuCheckpointProcessRestore");
    p_unlock = dlsym(cuda_handle, "cuCheckpointProcessUnlock");
    p_getRestoreTid = dlsym(cuda_handle, "cuCheckpointProcessGetRestoreThreadId");
}

/* Send FDs over unix socket */
static int send_fds(int sock, int *fds, int nfds) {
    char buf[1] = {'F'};
    struct iovec iov = { .iov_base = buf, .iov_len = 1 };
    char cmsg_buf[CMSG_SPACE(sizeof(int) * MAX_FDS)];
    struct msghdr msg = {0};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
    memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * nfds);
    return sendmsg(sock, &msg, 0);
}

/* Receive FDs over unix socket */
static int recv_fds(int sock, int *fds, int max_fds) {
    char buf[1];
    struct iovec iov = { .iov_base = buf, .iov_len = 1 };
    char cmsg_buf[CMSG_SPACE(sizeof(int) * MAX_FDS)];
    struct msghdr msg = {0};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);
    int n = recvmsg(sock, &msg, 0);
    if (n <= 0) return -1;
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg) return 0;
    int nfds = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    if (nfds > max_fds) nfds = max_fds;
    memcpy(fds, CMSG_DATA(cmsg), sizeof(int) * nfds);
    return nfds;
}

/* Find nvidia device FDs in /proc/self/fd */
static int find_nvidia_fds(int *fds, int max) {
    int n = 0;
    char path[256], link[256];
    for (int fd = 0; fd < 1024 && n < max; fd++) {
        snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
        ssize_t len = readlink(path, link, sizeof(link)-1);
        if (len > 0) {
            link[len] = 0;
            if (strstr(link, "/dev/nvidia")) {
                fds[n++] = fd;
                fprintf(stderr, "  found nvidia fd %d -> %s\n", fd, link);
            }
        }
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s save|keep|restore\n", argv[0]); return 1; }
    unsetenv("CUDA_VISIBLE_DEVICES");

    if (strcmp(argv[1], "save") == 0) {
        nvmlInit();
        load_cuda();
        p_cuInit(0);
        int ndev; p_cuDeviceGetCount(&ndev);
        printf("pid=%d gpus=%d\n", getpid(), ndev);

        CUcontext ctx; p_cuDevicePrimaryCtxRetain(&ctx, 0); p_cuCtxSetCurrent(ctx);
        CUdeviceptr dptr; p_cuMemAlloc(&dptr, 4*1024*1024);
        p_cuMemsetD32(dptr, 0xDEADBEEF, 1024*1024);
        printf("alloc+pattern on GPU 0\n");

        /* Lock + checkpoint */
        char a[64]; memset(a,0,64);
        int r = p_lock(getpid(), a);
        printf("lock=%d\n", r);
        memset(a,0,64);
        r = p_ckpt(getpid(), a);
        printf("checkpoint=%d\n", r);

        int restore_tid = -1;
        if (p_getRestoreTid) {
            p_getRestoreTid(getpid(), &restore_tid);
            printf("restore_tid=%d\n", restore_tid);
            FILE *f = fopen("/tmp/.restore_tid", "w");
            if (f) { fprintf(f, "%d\n", restore_tid); fclose(f); }
        }

        /* Find nvidia FDs */
        int nvidia_fds[MAX_FDS];
        fprintf(stderr, "Looking for nvidia FDs:\n");
        int nfds = find_nvidia_fds(nvidia_fds, MAX_FDS);
        printf("found %d nvidia FDs\n", nfds);

        /* Save FD count */
        FILE *f = fopen("/tmp/.nvidia_fd_count", "w");
        if (f) { fprintf(f, "%d\n", nfds); fclose(f); }

        /* Connect to keeper and send FDs */
        int sock = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr = {0};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path)-1);
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("connect to keeper");
            return 1;
        }
        send_fds(sock, nvidia_fds, nfds);
        printf("sent %d FDs to keeper\n", nfds);
        close(sock);

        printf("EXIT (checkpointed, FDs passed to keeper)\n");
        /* Close cuda handle but FDs survive in keeper */
        return 0;

    } else if (strcmp(argv[1], "keep") == 0) {
        /* Keeper: listen on unix socket, receive FDs, hold them */
        unlink(SOCK_PATH);
        int srv = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr = {0};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path)-1);
        bind(srv, (struct sockaddr*)&addr, sizeof(addr));
        listen(srv, 2);
        printf("keeper pid=%d listening\n", getpid());

        /* Receive FDs from saver */
        int client = accept(srv, NULL, NULL);
        int fds[MAX_FDS];
        int nfds = recv_fds(client, fds, MAX_FDS);
        close(client);
        printf("keeper: received %d FDs, holding them alive\n", nfds);
        for (int i = 0; i < nfds; i++) {
            char path[256], link[256];
            snprintf(path, sizeof(path), "/proc/self/fd/%d", fds[i]);
            ssize_t len = readlink(path, link, sizeof(link)-1);
            if (len > 0) { link[len] = 0; printf("  fd %d -> %s\n", fds[i], link); }
        }
        fflush(stdout);

        /* Wait for restore process to connect */
        int client2 = accept(srv, NULL, NULL);
        printf("keeper: restore process connected, sending FDs\n");
        send_fds(client2, fds, nfds);
        close(client2);

        /* Keep running briefly so FDs stay alive during restore */
        sleep(30);
        printf("keeper: exiting\n");
        return 0;

    } else if (strcmp(argv[1], "restore") == 0) {
        /* DON'T call cuInit yet — we want to use the passed FDs */
        printf("restore pid=%d\n", getpid());

        /* Connect to keeper and receive FDs */
        int sock = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr = {0};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path)-1);
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("connect to keeper");
            return 1;
        }
        int fds[MAX_FDS];
        int nfds = recv_fds(sock, fds, MAX_FDS);
        close(sock);
        printf("received %d FDs from keeper\n", nfds);
        for (int i = 0; i < nfds; i++) {
            char path[256], link[256];
            snprintf(path, sizeof(path), "/proc/self/fd/%d", fds[i]);
            ssize_t len = readlink(path, link, sizeof(link)-1);
            if (len > 0) { link[len] = 0; printf("  fd %d -> %s\n", fds[i], link); }
        }

        /* Now init CUDA — it should find the existing driver session via the FDs */
        load_cuda();
        CUresult r = p_cuInit(0);
        printf("cuInit=%d\n", r);

        /* Try restore */
        char ra[64]; memset(ra, 0, 64);
        printf("restore(%d)...\n", getpid()); fflush(stdout);
        r = p_restore(getpid(), ra);
        printf("restore=%d\n", r);

        if (r == 0) {
            char ua[64]; memset(ua, 0, 64);
            r = p_unlock(getpid(), ua);
            printf("unlock=%d\n", r);

            /* Verify GPU memory */
            CUcontext ctx; p_cuDevicePrimaryCtxRetain(&ctx, 0); p_cuCtxSetCurrent(ctx);
            unsigned int val;
            CUdeviceptr dptr; p_cuMemAlloc(&dptr, 4);
            /* Can't easily get the old dptr, but if restore worked, contexts are live */
            printf("FD_PASS_RESTORE_OK!\n");
        } else {
            printf("FD_PASS_RESTORE_FAILED: %d\n", r);
        }
        return r;
    }
    return 1;
}
