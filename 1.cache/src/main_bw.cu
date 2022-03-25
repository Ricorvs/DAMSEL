#include <cuda.h>
#include "pchase.h"
#include <getopt.h>



#ifndef CACHE_POLICY
#define CACHE_POLICY .ca
#endif

#define STR(x) #x
#define XSTR(s) STR(s)

#define WARP_SIZE 32

#ifndef BIT_SIZE
#define BIT_SIZE 32
#endif

#if BIT_SIZE == 32
#define FTYPE float
#define ASM_TYPE f
#else
#define FTYPE double
#define ASM_TYPE d
#endif

#define ADD 0
#define NEG 1
#define MOV 2
#define NONE 3

#if INSTR == ADD
#define INSTRUCTION "add.f" XSTR(BIT_SIZE) " %0, data, %0;\n\t"
#elif INSTR == NEG
#define INSTRUCTION "neg.f" XSTR(BIT_SIZE) " %0, data;\n\t"
#elif INSTR == MOV
#define INSTRUCTION "mov.f" XSTR(BIT_SIZE) " %0, data;\n\t"
#else
#define INSTRUCTION
#endif

char *fname = NULL;

static __device__ __inline__ uint32_t __mysmid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

/* https://arxiv.org/pdf/1804.06826.pdf */
__global__ void l1_bw(uint32_t num_threads, uint32_t arr_size,
                      uint32_t *startClk, uint32_t *stopClk,
                      FTYPE *dsink, FTYPE *posArray,
                      uint32_t *smid, uint32_t *bid, uint32_t *wid,
                      uint8_t *colmap) {
    posArray += blockIdx.x * arr_size;
    // thread index
    uint32_t tid = threadIdx.x;

    // a register to avoid compiler optimization
    FTYPE sink = 0;
    uint32_t start = 0;
    uint32_t stop = 0;

    if (tid < num_threads) {
        uint32_t i = blockIdx.x * num_threads + tid;
        // populate l1 cache to warm up
        for(uint32_t i = tid; i < arr_size; i += num_threads) {
            FTYPE *ptr = posArray + i;
            asm volatile ("{\t\n"
                ".reg .f" XSTR(BIT_SIZE) " data;\n\t"
                "ld.global" XSTR(CACHE_POLICY) ".f" XSTR(BIT_SIZE) " data, [%1];\n\t"
                INSTRUCTION
                "}" : "+" XSTR(ASM_TYPE) (sink) : "l"(ptr) : "memory"
            );
        }
        // synchronize all threads
        asm volatile ("bar.sync 0;");
        // start timing

        asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
        // load data from l1 cache and accumulate
        for(uint32_t i = 0; i < arr_size; i += num_threads) {
            FTYPE *ptr = posArray + i;
            // every warp loads all data in l1 cache
            for (uint32_t j = 0; j < num_threads; j += WARP_SIZE) {
                uint32_t offset = (tid + j) % num_threads;
                asm volatile ("{\t\n"
                    ".reg .f" XSTR(BIT_SIZE) " data;\n\t"
                    "ld.global" XSTR(CACHE_POLICY) ".f" XSTR(BIT_SIZE) " data, [%1];\n\t"
                    INSTRUCTION
                    "}" : "+" XSTR(ASM_TYPE) (sink) : "l"(ptr + offset) : "memory"
                );
            }
        }
        // synchronize all threads
        asm volatile ("bar.sync 0;");
        // stop timing
        asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
        // write time and data back to memory
        startClk[i] = start;
        stopClk[i] = stop;
        dsink[i] = sink;
        smid[i] = __mysmid();
        bid[i] = blockIdx.x;
        wid[i] = tid / WARP_SIZE;
        colmap[i] = __mysmid() == 0 && tid % WARP_SIZE == 0;
    }
    __syncthreads();
}

uint32_t run_l1_bw(uint32_t blocks, uint32_t warps, uint32_t active, uint32_t arr_size_b) {
    uint32_t *dev_startclk, *dev_stopclk, *dev_smid, *dev_wid, *dev_bid;
    uint32_t *startclk, *stopclk, *smid, *wid, *bid;
    uint8_t *dev_colmap, *colmap;
    FTYPE *dev_sink, *dev_arr;
    FILE *f = NULL;
    cudaDeviceProp prop;
    checkCudaCall(cudaGetDeviceProperties(&prop, 0));

    if (fname) {
        f = fopen(fname, "w");
    }

    uint32_t actual_blocks = prop.multiProcessorCount * blocks;
    uint32_t num_threads = warps * WARP_SIZE;
    uint32_t num_active = active * WARP_SIZE;
    uint32_t buf_size = actual_blocks * num_active;
    startclk = (uint32_t*) malloc(buf_size * sizeof(uint32_t));
    stopclk = (uint32_t*) malloc(buf_size * sizeof(uint32_t));
    smid = (uint32_t*) malloc(buf_size * sizeof(uint32_t));
    wid = (uint32_t*) malloc(buf_size * sizeof(uint32_t));
    bid = (uint32_t*) malloc(buf_size * sizeof(uint32_t));
    colmap = (uint8_t*) malloc(buf_size * sizeof(uint8_t));
    checkCudaCall(cudaMalloc((void**) &dev_startclk, buf_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_stopclk, buf_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_smid, buf_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_wid, buf_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_bid, buf_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_sink, buf_size * sizeof(double)));
    checkCudaCall(cudaMalloc((void**) &dev_colmap, buf_size * sizeof(uint8_t)));
    checkCudaCall(cudaMalloc((void**) &dev_arr, actual_blocks * arr_size_b));

    l1_bw<<<actual_blocks, num_threads>>>(num_active, arr_size_b / sizeof(double),
                                          dev_startclk, dev_stopclk, dev_sink,
                                          dev_arr, dev_smid, dev_bid, dev_wid,
                                          dev_colmap);

    checkCudaCall(cudaDeviceSynchronize());
    checkCudaCall(cudaMemcpy(startclk, dev_startclk, buf_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(stopclk, dev_stopclk, buf_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(smid, dev_smid, buf_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(wid, dev_wid, buf_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(bid, dev_bid, buf_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(colmap, dev_colmap, buf_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    uint32_t min = ~0, max = 0;
    for (uint32_t i = 0; i < buf_size; i += active) {
        if (colmap[i]) {
            if (startclk[i] < min) {
                min = startclk[i];
            }
            if (stopclk[i] > max) {
                max = stopclk[i];
            }
        }
    }

    uint32_t prev_bid = ~0, actual_bid = 0;
    for (uint32_t i = 0; i < buf_size; i++) {
        if (colmap[i]) {
            if (bid[i] != prev_bid) {
                if (prev_bid != ~0) {
                    actual_bid++;
                }
                prev_bid = bid[i];
            }
            if (f) {
                fprintf(f, "%d %d %d %d %d %f\n", smid[i], actual_bid, wid[i],
                       startclk[i] - min, stopclk[i] - min,
                       (double) arr_size_b / (stopclk[i] - startclk[i]));
            }
        }
    }

    printf("%d %d %d %d %d %f\n", arr_size_b, blocks, active, warps, max - min, (double) (blocks * active * arr_size_b) / (max - min));

    checkCudaCall(cudaFree(dev_startclk));
    checkCudaCall(cudaFree(dev_stopclk));
    checkCudaCall(cudaFree(dev_smid));
    checkCudaCall(cudaFree(dev_wid));
    checkCudaCall(cudaFree(dev_bid));
    checkCudaCall(cudaFree(dev_sink));
    checkCudaCall(cudaFree(dev_arr));
    checkCudaCall(cudaFree(dev_colmap));

    free(startclk);
    free(stopclk);
    free(smid);
    free(wid);
    free(bid);
    free(colmap);

    if (f) {
        fclose(f);
    }

    return 0;
}

static void usage(const char *pname)
{
    printf("Usage: %s [OPTION]...\n"
           "\n"
           "  -n NUM     Set the array size (in Bytes).\n"
           "  -w NUM     Set number of warps to use.\n"
           "  -b NUM     Set number of blocks to use.\n"
           "  -a NUM     Set number of active blocks.\n"
           "  -1         Run L1 test.\n"
           "  -?         Print this help.\n"
           ,pname);
    exit(0);
}

int main(int argc, char *argv[]) {
    int ch;
    uint32_t warps = 32;
    uint32_t active = 32;
    uint32_t arr_size = 24 * 1024;
    uint32_t blocks = 1;
    uint32_t cache_level = 1;

    while ((ch = getopt(argc, argv, "f:b:w:a:n:12?")) != -1)
    {
        switch(ch) {
        case 'w': warps = strtol(optarg, 0, 10); break;
        case 'b': blocks = strtol(optarg, 0, 10); break;
        case 'a': active = strtol(optarg, 0, 10); break;
        case 'n': arr_size = strtol(optarg, 0, 10); break;
        case 'f': fname = optarg; break;
        case '1': cache_level = 1; break;
        case '?': default: usage(argv[0]);
        }
    }

    if (cache_level == 1) {
        run_l1_bw(blocks, warps, active, arr_size);
    }
    return 0;
}
