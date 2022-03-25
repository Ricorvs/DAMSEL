#include "pchase.h"
#include "stdio.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cooperative_groups.h>

#define MASK 0x00000000FFFFFFFF
#ifndef CACHE_POLICY
#define CACHE_POLICY .ca
#endif

#define STR(x) #x
#define XSTR(s) STR(s)

namespace cg = cooperative_groups;

static __device__ __inline__ uint32_t __mysmid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

void calculate_stats(uint32_t *values, uint32_t n, uint32_t threshold, stats_t *stats) {
    stats->avg = 0;
    stats->min = ~0;
    stats->max = 0;
    stats->hits = 0, stats->miss = 0;
    for (int i = 0; i < n; i++) {
        stats->avg += values[i];

        if (values[i] < stats->min) {
            stats->min = values[i];
        }
        if (values[i] > stats->max) {
            stats->max = values[i];
        }
        if (values[i] < threshold) {
            stats->hits++;
        } else {
            stats->miss++;
        }
    }
    stats->avg /= n;
}

void calculate_median(stats_t *stats, uint32_t n, stats_t *out) {
    uint32_t buf[n];

    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < n; j++) {
            buf[j] = *((uint32_t*) (stats + j) + i);
        }
        qsort(buf, n, sizeof(uint32_t), compare_uint);
        uint32_t median = (buf[n / 2] + buf[n / 2 - ((n & 1) ^ 1)]) / 2;
        *((uint32_t*) out + i) = median;
    }
}

int compare_uint(const void *a,const void *b) {
    uint32_t *x = (uint32_t *) a;
    uint32_t *y = (uint32_t *) b;
    return *x - *y;
}

__global__ void init_array(uint32_t *A, int arr_size, int stride) {
    for (uint32_t i = 0; i < arr_size; i++) {
        A[i] = (i + stride) % arr_size;
    }
}

__global__ void pchase(uint32_t *A, int preheat, int arr_size, int iterations, uint32_t *times) {
    extern __shared__ unsigned int s_tvalue[];
    extern __shared__ unsigned int s_index[];
    int i, j = 0;
    clock_t start, end;

    if (preheat) {
        for (i = 0; i < arr_size; i++) {
            asm volatile ("{\t\n"
                "ld.global" XSTR(CACHE_POLICY) ".u32 %0, [%1];\n\t"
                "}" : "=r"(j): "l"(A + i) : "memory"
            );
            s_index[i % iterations] = j;
        }
    }
    j = 0;

    for (i = 0; i < iterations; i++) {
        start = clock();
        asm volatile ("{\t\n"
            "ld.global" XSTR(CACHE_POLICY) ".u32 %0, [%1];\n\t"
            "}" : "=r"(j): "l"(A + j) : "memory"
        );
        s_index[i] = j;
        end = clock();
        s_tvalue[i] = end - start;
    }

    for (i = 0; i < iterations; i++) {
        times[i] = s_tvalue[i];
    }
}

texture<uint32_t, 1, cudaReadModeElementType> tex1Ref;
__global__ void pchase_texture(int preheat, int arr_size, int iterations, uint32_t *times) {
    extern __shared__ unsigned int s_tvalue[];
    extern __shared__ unsigned int s_index[];
    int i, j = 0;
    clock_t start, end;

    if (preheat) {
        for (i = 0; i < arr_size; i++) {
            s_index[i % iterations] = tex1Dfetch<uint32_t>(tex1Ref, i);
        }
    }
    j = 0;

    for (i = 0; i < iterations; i++) {
        start = clock();
        j = tex1Dfetch<uint32_t>(tex1Ref, j);
        s_index[i] = j;
        end = clock();
        s_tvalue[i] = end - start;
    }

    for (i = 0; i < iterations; i++) {
        times[i] = s_tvalue[i];
    }
}

__global__ void pchase_multi_sm_times(uint32_t *A, int arr_size, int iterations,
                                      uint32_t preheat, uint8_t *colmap, uint32_t *times) {
    extern __shared__ uint64_t s_tvalue64[];
    extern __shared__ uint64_t s_index64[];
    clock_t start, end;
    uint32_t *arr = A + blockIdx.x * arr_size;
    int i, j = 0;

    __syncthreads();

    if (preheat) {
        for (i = 0; i < arr_size; i++) {
            asm volatile ("{\t\n"
                "ld.global" XSTR(CACHE_POLICY) ".u32 %0, [%1];\n\t"
                "}" : "=r"(j): "l"(arr + i) : "memory"
            );
            s_index64[i % iterations] = j;
        }
    }
    j = arr[0];

    __syncthreads();

    if (__mysmid() == 0) {
        for (i = 0; i < iterations; i++) {
            start = clock();
            asm volatile ("{\t\n"
                "ld.global" XSTR(CACHE_POLICY) ".u32 %0, [%1];\n\t"
                "}" : "=r"(j): "l"(arr + j) : "memory"
            );
            *((uint32_t*) (s_index64 + i)) = j;
            end = clock();
            s_tvalue64[i] = ((end & MASK) << 32) | (start & MASK);
        }
    }
    __syncthreads();
    if (__mysmid() == 0) {
        uint32_t *tarr = (uint32_t*) ((uint64_t*) times + blockIdx.x * iterations);
        for (i = 0; i < iterations; i++) {
            tarr[i] = *((uint32_t*) (s_tvalue64 + i));
            tarr[i + iterations] = *((uint32_t*) (s_tvalue64 + i) + 1);
        }
    }
    colmap[blockIdx.x * 2] = __mysmid() == 0;
    colmap[blockIdx.x * 2 + 1] = __mysmid() == 0;
}

__global__ void pchase_sm(uint32_t *A, int arr_size, int iterations,
                          uint32_t heatwarp, uint32_t loadwarp,
                          uint32_t maxlane, uint32_t *times) {
    extern __shared__ uint64_t s_tvalue64[];
    extern __shared__ uint64_t s_index64[];
    int i, j = 0;

    s_index64[0] = 0;
    uint32_t laneid = threadIdx.x & (32 - 1);
    uint32_t warpid = threadIdx.x / 32;

    __syncthreads();
    clock_t start, end;

    if (laneid == 0 && (warpid == heatwarp || heatwarp == ~0)) {
        for (i = 0; i < arr_size; i++) {
            asm volatile ("{\t\n"
                "ld.global" XSTR(CACHE_POLICY) ".u32 %0, [%1];\n\t"
                "}" : "=r"(j): "l"(A + i) : "memory"
            );
            s_index64[i % iterations] = j;
        }
    }
    j = laneid;

    int offset = (loadwarp == ~0) ? warpid * maxlane * iterations : 0;
    if (laneid < maxlane) {
        offset += laneid * iterations;
    }
    __syncthreads();

    if (laneid < maxlane && (warpid == loadwarp || loadwarp == ~0)) {
        for (i = 0; i < iterations; i++) {
            start = clock();
            asm volatile ("{\t\n"
                "ld.global" XSTR(CACHE_POLICY) ".u32 %0, [%1];\n\t"
                "}" : "=r"(j): "l"(A + j) : "memory"
            );
            *((uint32_t*) (s_index64 + i + offset)) = j;
            end = clock();
            s_tvalue64[i + offset] = ((end & MASK) << 32) | (start & MASK);
        }
    }
    __syncthreads();

    if (laneid < maxlane && (warpid == loadwarp || loadwarp == ~0)) {
        uint32_t *tarr = (uint32_t*) ((uint64_t*) times + offset);
        uint64_t *source = s_tvalue64 + offset;
        for (i = 0; i < iterations; i++) {
            tarr[i] = (uint32_t) ((source[i] << 32) >> 32);
            tarr[i + iterations] = (uint32_t) (source[i] >> 32);
        }
    }
}

void pchase_texture_body(pchase_args_t *args, void *eargs) {
    uint32_t *dev_times, *dev_arr;
    checkCudaCall(cudaMalloc((void**) &dev_arr, args->arr_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_times, args->iters * sizeof(uint32_t)));
    init_array<<<1, 1>>>(dev_arr, args->arr_size, args->stride);

    cudaChannelFormatDesc texdesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    // Create texture object
    size_t offset = 0;
    checkCudaCall(cudaBindTexture(&offset, &tex1Ref, dev_arr, &texdesc,
                                  args->arr_size * sizeof(uint32_t)));

    pchase_texture<<<1, 1, args->iters * sizeof(uint32_t)>>>(args->preheat,
                                                             args->arr_size,
                                                             args->iters,
                                                             dev_times);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaMemcpy(args->ret_arr,
                             dev_times, args->iters * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(dev_arr));
    checkCudaCall(cudaFree(dev_times));
}

void pchase_body(pchase_args_t *args, void *eargs) {
    uint32_t *dev_times, *dev_arr;
    checkCudaCall(cudaMalloc((void**) &dev_arr, args->arr_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_times, args->iters * sizeof(uint32_t)));

    init_array<<<1, 1>>>(dev_arr, args->arr_size, args->stride);
    pchase<<<1, 1, args->iters * sizeof(uint32_t)>>>(dev_arr,
                                                     args->preheat,
                                                     args->arr_size,
                                                     args->iters,
                                                     dev_times);
    checkCudaCall(cudaDeviceSynchronize());
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaMemcpy(args->ret_arr, dev_times,
                             args->iters * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(dev_arr));
    checkCudaCall(cudaFree(dev_times));
}

void pchase_multi_sm_body(pchase_args_t *args, void *eargs) {
    pchase_sm_args_t *sm_args = (pchase_sm_args_t*) eargs;
    uint32_t *dev_arr, *dev_time;
    uint8_t *dev_colmap;
    uint32_t blocks = sm_args->blocks;
    uint32_t ret_size = blocks * args->iters * 2;
    sm_args->colmap = (uint8_t*) calloc(blocks * 2, sizeof(uint8_t));

    checkCudaCall(cudaMalloc((void**) &dev_arr, blocks * args->arr_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_colmap, blocks * 2 * sizeof(uint8_t)));
    checkCudaCall(cudaMalloc((void**) &dev_time, ret_size * sizeof(uint32_t)));
    for (int i = 0; i < blocks; i++) {
        init_array<<<1, 1>>>(dev_arr + args->arr_size * i, args->arr_size, args->stride);
    }
    pchase_multi_sm_times<<<blocks, 1, 2 * args->iters * sizeof(uint32_t)>>>
        (dev_arr, args->arr_size, args->iters, args->preheat, dev_colmap, dev_time);

    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaMemcpy(args->ret_arr, dev_time, ret_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(sm_args->colmap, dev_colmap, blocks * 2 * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(dev_arr));
    checkCudaCall(cudaFree(dev_time));

    uint32_t minimum = ~0;
    for (int j = 0; j < blocks; j++) {
        if (!sm_args->colmap[j * 2]) {
            continue;
        }
        if (args->ret_arr[j * args->iters * 2] < minimum) {
            minimum = args->ret_arr[j * args->iters * 2];
        }
    }
    for (int j = 0; j < ret_size; j++) {
        if (args->ret_arr[j] != 0) {
            args->ret_arr[j] -= minimum;
        }
    }
}


void pchase_sm_body(pchase_args_t *args, void *eargs) {
    pchase_sm_args_t *sm_args = (pchase_sm_args_t*) eargs;
    uint32_t *dev_arr;
    uint32_t *dev_times;
    uint32_t blocks = sm_args->blocks;
    uint32_t ret_size = sm_args->laneid * blocks * args->iters * 2;

    checkCudaCall(cudaMalloc((void**) &dev_arr, args->arr_size * sizeof(uint32_t)));
    checkCudaCall(cudaMalloc((void**) &dev_times, ret_size * sizeof(uint32_t)));
    init_array<<<1, 1>>>(dev_arr, args->arr_size, args->stride);

    pchase_sm<<<1, 32 * blocks, ret_size * sizeof(uint32_t)>>>
        (dev_arr, args->arr_size, args->iters, sm_args->heatwarp, sm_args->loadwarp, sm_args->laneid, dev_times);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaMemcpy(args->ret_arr, dev_times, ret_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(dev_arr));
    checkCudaCall(cudaFree(dev_times));

    uint32_t minimum = ~0;
    for (int j = 0; j < blocks * sm_args->laneid; j++) {
        if (args->ret_arr[j * args->iters * 2] < minimum) {
            minimum = args->ret_arr[j * args->iters * 2];
        }
    }
    for (int j = 0; j < ret_size; j++) {
        args->ret_arr[j] -= minimum;
    }
}

void run_pchase(pchase_args_t *args, stats_t *stats, void *eargs) {
    uint32_t iters = args->iters;
    uint32_t repeat = args->repeat;
    uint32_t hit_threshold = args->hit_threshold;

    uint32_t *ret_arr;
    uint32_t arr_size = iters;

    if (args->ret_arr_size != 0) {
        ret_arr = (uint32_t*) malloc(repeat * args->ret_arr_size * sizeof(uint32_t));
        arr_size = args->ret_arr_size;
    } else {
        ret_arr = (uint32_t*) malloc(repeat * arr_size * sizeof(uint32_t));
    }
    stats_t r_stats[repeat];

    for (int r = 0; r < repeat; r++) {
        args->ret_arr = ret_arr + r * arr_size;
        args->body(args, eargs);
        calculate_stats(args->ret_arr, arr_size, hit_threshold, r_stats + r);
    }

    args->ret_arr = ret_arr;

    calculate_median(r_stats, repeat, stats);
}

void export_to_file(uint32_t *arr, uint32_t repeat, uint32_t arr_size,
                    uint32_t cols, uint32_t iters, uint32_t stride,
                    uint8_t *colmap, char *fname) {
    FILE *outfile = fopen(fname, "w");
    uint32_t buf[repeat];
    uint32_t col_offset = cols * iters;

    for (int i = 0; i < iters; i++) {
        fprintf(outfile, "%u %u ", i, i * stride % arr_size * sizeof(uint32_t));
        for (int j = 0; j < cols; j++) {
            if (colmap && !colmap[j]) {
                continue;
            }
            uint32_t arr_offset = iters * j;
            if (repeat == 1) {
                fprintf(outfile, "%u ", arr[i + arr_offset]);
            } else {
                for (int k = 0; k < repeat; k++) {
                    buf[j] = arr[i + arr_offset + k * col_offset];
                }
                qsort(buf, repeat, sizeof(uint32_t), compare_uint);
                fprintf(outfile, "%u ", (uint32_t) ((double) (buf[repeat / 2] + buf[repeat / 2 - (repeat & 1)]) / 2));
            }
        }
        fprintf(outfile, "\n");
    }
    fclose(outfile);
}
