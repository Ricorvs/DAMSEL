#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <getopt.h>
#include <stdio.h>

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600)
#error "All configurations require CC level >= 6.0"
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700) && (CONFIG <= 5)
#error "Configurations 0 through 5 require CC level >= 7.0"
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) && (CONFIG >= 6 && CONFIG <= 16)
#error "Configurations 6 through 16 require CC level >= 8.0"
#endif

#if CONFIG == 15
#error "Configuration 15 is currently not functional"
#endif

#ifndef __HALF_TO_US
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#endif
#ifndef __HALF2_TO_UI
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#endif

#define WARP_SIZE 32

#ifndef CONFIG
#define CONFIG 0
#endif


#if CONFIG < 3 // half * half = float
    #define MATTYPE_AB half
    #define MATTYPE_C float
    #define MATK 16
    #if CONFIG == 0
        #define MATM 16
        #define MATN 16
    #elif CONFIG == 1
        #define MATM 32
        #define MATN 8
    #else
        #define MATM 8
        #define MATN 32
    #endif
#elif CONFIG < 6 // half * half = half
    #define MATTYPE_AB half
    #define MATTYPE_C half
    #define MATK 16
    #if CONFIG == 3
        #define MATM 16
        #define MATN 16
    #elif CONFIG == 4
        #define MATM 32
        #define MATN 8
    #else
        #define MATM 8
        #define MATN 32
    #endif
#elif CONFIG < 9 // uchar * uchar = int
    #define MATTYPE_AB unsigned char
    #define MATTYPE_C int
    #define MATK 16
    #if CONFIG == 6
        #define MATM 16
        #define MATN 16
    #elif CONFIG == 7
        #define MATM 32
        #define MATN 8
    #else
        #define MATM 8
        #define MATN 32
    #endif
#elif CONFIG < 12 // char * char = int
    #define MATTYPE_AB signed char
    #define MATTYPE_C int
    #define MATK 16
    #if CONFIG == 9
        #define MATM 16
        #define MATN 16
    #elif CONFIG == 10
        #define MATM 32
        #define MATN 8
    #else
        #define MATM 8
        #define MATN 32
    #endif
#elif CONFIG < 15 // bf16 * bf16 = float
    #define MATTYPE_AB __nv_bfloat16
    #define MATTYPE_C float
    #define MATK 16
    #if CONFIG == 12
        #define MATM 16
        #define MATN 16
    #elif CONFIG == 13
        #define MATM 32
        #define MATN 8
    #else
        #define MATM 8
        #define MATN 32
    #endif
#elif CONFIG == 15 // tf32 * tf32 = flaot
    #define MATTYPE_AB precision::tf32
    #define MATTYPE_C float
    #define MATK 8
    #define MATM 16
    #define MATN 16
#elif CONFIG == 16 // double * double = double
    #define MATTYPE_AB double
    #define MATTYPE_C double
    #define MATK 4
    #define MATM 8
    #define MATN 8
#elif CONFIG == 17
    #define MATTYPE_AB half
    #define MATTYPE_C float
    #define TC 0
    #define MATK 16
    #define MATM 4
    #define MATN 8
#elif CONFIG == 18
    #define MATTYPE_AB half
    #define MATTYPE_C half
    #define TC 0
    #define MATK 16
    #define MATM 4
    #define MATN 8
#elif CONFIG == 19
    #define MATTYPE_AB unsigned char
    #define MATTYPE_C int
    #define TC 0
    #define MATK 16
    #define MATM 4
    #define MATN 8
#elif CONFIG == 20
    #define MATTYPE_AB double
    #define MATTYPE_C double
    #define TC 0
    #define MATK 4
    #define MATM 4
    #define MATN 8
#elif CONFIG == 21
    #define MATTYPE_AB float
    #define MATTYPE_C float
    #define TC 0
    #define MATK 16
    #define MATM 4
    #define MATN 8
#endif

#ifndef TC
#define TC 1
#endif

char *dir = NULL;

#ifdef TIMING
static __device__ __inline__ uint32_t __mysmid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}
#endif

static void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr,"checkCuda: %s %s %d\n", cudaGetErrorString(result), file, line);
        exit(1);
    }
}

#define checkCudaCall(res) { checkCuda((res), __FILE__, __LINE__); }

__global__ void init_mat(MATTYPE_AB *A, uint32_t num_threads, uint32_t lim) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < lim; i += num_threads) {
        A[i] = i % 16;
    }
}

#if TC == 1
#if ((__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4) || (__CUDACC_VER_MAJOR__ > 11))
using namespace ::nvcuda;
#endif

#ifdef TIMING
__global__ void tc_matmult(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c,
                           uint32_t m, uint32_t n, uint32_t k,
                           uint64_t *startclk, uint64_t *stopclk, uint8_t *colmap) {
    uint64_t start, end;
#else
__global__ void __launch_bounds__(1024, 1) tc_matmult(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c,
                           uint32_t m, uint32_t n, uint32_t k) {
#endif
    uint32_t warps = gridDim.x * (blockDim.x / warpSize);
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp = idx / warpSize;
    uint32_t mtiles = m / MATM;
    uint32_t ntiles = n / MATN;
    uint32_t tot_tiles = mtiles * ntiles;
    uint32_t ktiles = k / MATK;

    wmma::fragment<wmma::matrix_a, MATM, MATN, MATK, MATTYPE_AB, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, MATM, MATN, MATK, MATTYPE_AB, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, MATM, MATN, MATK, MATTYPE_C> c_frag;


    #ifdef TIMING
    start = clock64();
    #endif

    for (uint32_t i = warp; i < tot_tiles; i += warps) {
        uint32_t mtile = i / ntiles;
        uint32_t ntile = i % ntiles;
        MATTYPE_AB *a_offset = a + mtile * MATM * k;
        MATTYPE_AB *b_offset = b + ntile * MATN;
        wmma::fill_fragment(c_frag, 0.0);

        for (uint32_t j = 0; j < ktiles; j++, a_offset += MATK, b_offset += MATK * n) {
            wmma::load_matrix_sync(a_frag, a_offset, k);
            wmma::load_matrix_sync(b_frag, b_offset, n);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c + ntile * MATN + mtile * MATM * n, c_frag, n, wmma::mem_row_major);
    }
    #ifdef TIMING
    end = clock64();
    if (threadIdx.x % warpSize == 0) {
        startclk[warp] = start;
        stopclk[warp] = end;
        colmap[warp] = __mysmid() == 0;
    }
    #endif
}
#endif

#ifdef TIMING
__global__ void cc_matmult(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c,
                           uint32_t m, uint32_t n, uint32_t k,
                           uint64_t *startclk, uint64_t *stopclk, uint8_t *colmap) {
    uint64_t start, end;
#else
__global__ void cc_matmult(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c,
                           uint32_t m, uint32_t n, uint32_t k) {
#endif
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threads = gridDim.x * blockDim.x;
    uint32_t lim = m * n;

    #ifdef TIMING
    uint32_t warp = idx / warpSize;
    start = clock64();
    #endif


    for (uint32_t i = idx; i < lim; i += threads) {
        MATTYPE_C accumulator = 0;
        MATTYPE_AB *a_offset = a + i / n * k;
        MATTYPE_AB *b_offset = b + i % n;

        for (uint32_t j = 0; j < k; j++, a_offset++, b_offset += n) {
            accumulator += (MATTYPE_C) (*a_offset * *b_offset);
        }

        c[i] = accumulator;
    }

    #ifdef TIMING
    end = clock64();
    if (threadIdx.x % warpSize == 0) {
        startclk[warp] = start;
        stopclk[warp] = end;
        colmap[warp] = __mysmid() == 0;
    }
    #endif
}

double run_matmult(size_t warps, size_t blocks, uint32_t matn, uint32_t matm,
                   uint32_t matk, double *flops, uint64_t *cycles,
                   double *cycle_flops, int exp, uint32_t check) {
    MATTYPE_AB *dev_a, *dev_b;
    MATTYPE_C *c, *c2, *dev_c;
    uint64_t *startclk, *dev_startclk, *stopclk, *dev_stopclk;
    uint8_t *colmap, *dev_colmap;
    size_t threads = warps * WARP_SIZE;
    size_t tot_warps = warps * blocks;
    struct timespec start, end;
    uint32_t matsize_A = matm * matk;
    uint32_t matsize_B = matn * matk;
    uint32_t matsize_C = matn * matm;
    checkCudaCall(cudaMalloc((void**) &dev_a, matsize_A * sizeof(MATTYPE_AB)));
    checkCudaCall(cudaMalloc((void**) &dev_b, matsize_B * sizeof(MATTYPE_AB)));
    checkCudaCall(cudaMalloc((void**) &dev_c, matsize_C * sizeof(MATTYPE_C)));
    assert(((unsigned long long)dev_a) % 128 == 0);
    assert(((unsigned long long)dev_b) % 128 == 0);
    assert(((unsigned long long)dev_c) % 128 == 0);
    if (check) {
        c2 = (MATTYPE_C*) malloc(matsize_C * sizeof(MATTYPE_C));
    }
    checkCudaCall(cudaMalloc((void**) &dev_startclk, tot_warps * sizeof(uint64_t)));
    checkCudaCall(cudaMalloc((void**) &dev_stopclk, tot_warps * sizeof(uint64_t)));
    checkCudaCall(cudaMalloc((void**) &dev_colmap, tot_warps * sizeof(uint8_t)));

    c = (MATTYPE_C*) malloc(matsize_C * sizeof(MATTYPE_C));
    startclk = (uint64_t*) malloc(tot_warps * sizeof(uint64_t));
    stopclk = (uint64_t*) malloc(tot_warps * sizeof(uint64_t));
    colmap = (uint8_t*) malloc(tot_warps * sizeof(uint8_t));

    init_mat<<<1, 256>>>(dev_a, 256, matsize_A);
    init_mat<<<1, 256>>>(dev_b, 256, matsize_B);
    checkCudaCall(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &start);
    #if TC == 1
    #ifdef TIMING
    tc_matmult<<<blocks, threads>>>(dev_a, dev_b, dev_c, matm, matn, matk, dev_startclk, dev_stopclk, dev_colmap);
    #else
    tc_matmult<<<blocks, threads>>>(dev_a, dev_b, dev_c, matm, matn, matk);
    #endif
#else
    #ifdef TIMING
    cc_matmult<<<blocks, threads>>>(dev_a, dev_b, dev_c, matm, matn, matk, dev_startclk, dev_stopclk, dev_colmap);
    #else
    cc_matmult<<<blocks, threads>>>(dev_a, dev_b, dev_c, matm, matn, matk);
    #endif
#endif
    checkCudaCall(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    checkCudaCall(cudaGetLastError());

    checkCudaCall(cudaMemcpy(c, dev_c, matsize_C * sizeof(MATTYPE_C), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(startclk, dev_startclk, tot_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(stopclk, dev_stopclk, tot_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(colmap, dev_colmap, tot_warps * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // if (check) {
    //     printf("Check\n");
    // #ifdef TIMING
    //     cc_matmult<<<blocks, threads>>>(dev_a, dev_b, dev_c, matm, matn, matk, dev_startclk, dev_stopclk, dev_colmap);
    // #else
    //     cc_matmult<<<blocks, threads>>>(dev_a, dev_b, dev_c, matm, matn, matk);
    // #endif
    //     checkCudaCall(cudaDeviceSynchronize());
    //     checkCudaCall(cudaMemcpy(c2, dev_c, matsize_C * sizeof(MATTYPE_C), cudaMemcpyDeviceToHost));
    //     for (uint32_t i = 0; i < matsize_C; i++) {
    //         if (c[i] != c2[i]) {
    //             printf("%d %f %f\n", i, c[i], c2[i]);
    //         }
    //     }
    // }

#ifdef TIMING
    uint64_t min = ~0, max = 0;
    for (uint32_t i = 0; i < tot_warps; i++) {
        if (colmap[i]) {
            if (startclk[i] < min) {
                min = startclk[i];
            }
            if (stopclk[i] > max) {
                max = stopclk[i];
            }
        }
    }

    if (exp) {
        char fname[1024];
        sprintf(fname, "%s/%02d_%d_%d_%d_%05d_%02d.csv", dir, CONFIG, matm, matn, matk, blocks, warps);
        FILE *f = fopen(fname, "w");
        for (uint32_t i = 0; i < tot_warps; i++) {
            printf("%d %d\n", i, colmap[i]);
            if (colmap[i]) {
                fprintf(f, "%d %lu %lu\n", i, startclk[i] - min, stopclk[i] - min);
            }
        }

        fclose(f);
    }
#endif
    size_t ops = (size_t) matn * matm * matk * 2;

    double t = (end.tv_sec + (end.tv_nsec / 1000000000.0)) -
               (start.tv_sec + (start.tv_nsec / 1000000000.0));
    if (flops != NULL) {
        *flops = (double) ops / t;
    }

#ifdef TIMING
    if (cycles != NULL) {
        *cycles = max - min;
    }

    if (cycle_flops != NULL) {
        *cycle_flops = (double) ops / (max - min);
    }
#endif
    free(c);
    free(startclk);
    free(stopclk);
    free(colmap);
    checkCudaCall(cudaFree(dev_a));
    checkCudaCall(cudaFree(dev_b));
    checkCudaCall(cudaFree(dev_c));
    checkCudaCall(cudaFree(dev_startclk));
    checkCudaCall(cudaFree(dev_stopclk));
    checkCudaCall(cudaFree(dev_colmap));
    if (check) {
        free(c2);
    }
    return t;
}

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    checkCudaCall(cudaGetDeviceProperties(&prop, 0));
    int ch;
    int min_warps = -1;
    int max_warps = 32;
    int min_blocks = -1;
    int max_blocks = 1;
    uint64_t cycles;
    uint64_t matm, matn, matk;
    uint32_t mode = 0;
    uint32_t exp = 0;
    uint32_t check = 0;
    double flops, cycle_flops;

    while ((ch = getopt(argc, argv, "M:N:K:w:W:b:B:d:cm:e:")) != -1)
    {
        switch(ch) {
            case 'd': dir = optarg; break;
            case 'M': matm = strtol(optarg, 0, 10); break;
            case 'N': matn = strtol(optarg, 0, 10); break;
            case 'K': matk = strtol(optarg, 0, 10); break;
            case 'm': mode = strtol(optarg, 0, 10); break;
            case 'w': min_warps = strtol(optarg, 0, 10); break;
            case 'W': max_warps = strtol(optarg, 0, 10); break;
            case 'b': min_blocks = strtol(optarg, 0, 10); break;
            case 'B': max_blocks = strtol(optarg, 0, 10); break;
            case 'c': check = 1; break;
            case 'e': exp = 1; break;
        }
    }

    if (min_warps == -1) {
        min_warps = max_warps;
    }

    if (min_blocks == -1) {
        min_blocks = max_blocks;
    }

    for (int32_t blocks = min_blocks; blocks <= max_blocks; blocks++) {
        for (int32_t warps = min_warps; warps <= max_warps; warps++) {
            double t;
            if (mode == 0) {
                t = run_matmult(warps, blocks, matm, matn, matk, &flops, &cycles, &cycle_flops, exp, check);
            } else {
                t = run_matmult(warps, blocks * prop.multiProcessorCount, matm, matn, matk, &flops, &cycles, &cycle_flops, exp, check);
            }
            printf("%ld %ld %ld %d %d %.6e %.6e %lu %.6e\n", matm, matn, matk, warps, blocks, t, flops, cycles, cycle_flops);
        }
    }

}
