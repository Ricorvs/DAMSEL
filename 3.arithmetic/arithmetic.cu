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

#define STR(x) #x
#define XSTR(s) STR(s)

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
    #define MATTYPE_C __half2
    #define _INSTR fma
    #define ASM_TYPE f
    #define OP_TYPE r
    #define FUNC(x) __HALF2_TO_UI(x)
    #define BIT_SIZE 16x2
    #define TC 0
    #define MATK 32
    #define MATM 1
    #define MATN 1
#elif CONFIG == 18
    #define MATTYPE_AB half
    #define MATTYPE_C half
    #define _INSTR fma
    #define BIT_SIZE 16
    #define ASM_TYPE f
    #define OP_TYPE h
    #define FUNC(x) __HALF_TO_US(x)
    #define TC 0
    #define MATK 16
    #define MATM 1
    #define MATN 1
#elif CONFIG == 19
    #define MATTYPE_AB unsigned char
    #define MATTYPE_C int
    #define BIT_SIZE 32
    #define ASM_TYPE s
    #define OP_TYPE r
    #define ROUNDING .lo
    #define ROUNDING2
    #define TC 0
    #define MATK 16
    #define MATM 1
    #define MATN 1
#elif CONFIG == 20
    #define MATTYPE_AB double
    #define MATTYPE_C double
    #define ASM_TYPE f
    #define OP_TYPE d
    #define BIT_SIZE 64
    #define TC 0
    #define MATK 16
    #define MATM 1
    #define MATN 1
#elif CONFIG == 21
    #define MATTYPE_AB float
    #define MATTYPE_C float
    #define ASM_TYPE f
    #define OP_TYPE f
    #define BIT_SIZE 32
    #define TC 0
    #define MATK 16
    #define MATM 1
    #define MATN 1
#endif

#ifndef ROUNDING
#define ROUNDING .rn
#define ROUNDING2 .rn
#endif

#ifndef FUNC
#define FUNC(x) x
#endif

#ifndef _INSTR
#define _INSTR mad
#endif

#if INSTR == 0
#define INSTRUCTION XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %0, %8, %9, %0;\n\t" \
    XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %1, %8, %9, %1;\n\t" \
    XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %2, %8, %9, %2;\n\t" \
    XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %3, %8, %9, %3;\n\t" \
    XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %4, %8, %9, %4;\n\t" \
    XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %5, %8, %9, %5;\n\t" \
    XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %6, %8, %9, %6;\n\t" \
    XSTR(_INSTR) XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %7, %8, %9, %7;\n\t"
#elif INSTR == 1
#define INSTRUCTION ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data0;\n\t" \
    ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data1;\n\t" \
    ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data2;\n\t" \
    ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data3;\n\t" \
    ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data4;\n\t" \
    ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data5;\n\t" \
    ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data6;\n\t" \
    ".reg ." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data7;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data0, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %0, data0, %0;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data1, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %1, data1, %1;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data2, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %2, data2, %2;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data3, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %3, data3, %3;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data4, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %4, data4, %4;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data5, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %5, data5, %5;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data6, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %6, data6, %6;\n\t" \
    "mul" XSTR(ROUNDING) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " data7, %8, %9;\n\t" \
    "add" XSTR(ROUNDING2) "." XSTR(ASM_TYPE) XSTR(BIT_SIZE) " %7, data7, %7;\n\t"
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
        A[i] = i;
    }
}

#if TC == 1
#if ((__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4) || (__CUDACC_VER_MAJOR__ > 11))
using namespace ::nvcuda;
#endif

#ifdef TIMING
__global__ void repeat_tc(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c, int n,
                          uint64_t *startclk, uint64_t *stopclk, uint8_t *colmap) {
    uint64_t start, end;
#else
__global__ void repeat_tc(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c, int n) {
#endif
    uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    wmma::fragment<wmma::matrix_a, MATM, MATN, MATK, MATTYPE_AB, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, MATM, MATN, MATK, MATTYPE_AB, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, MATM, MATN, MATK, MATTYPE_C> c_frag;

    wmma::fill_fragment(c_frag, 0.0);
    wmma::load_matrix_sync(a_frag, a + warp * MATM * MATK, MATN);
    wmma::load_matrix_sync(b_frag, b + warp * MATN * MATK, MATM);
#ifdef TIMING
    start = clock64();
#endif
    for (int i = 0; i < n; i++) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
#ifdef TIMING
    end = clock64();
    if (threadIdx.x % warpSize == 0) {
        startclk[warp] = start;
        stopclk[warp] = end;
        colmap[warp] = __mysmid() == 0;
    }
#endif
    wmma::store_matrix_sync(c + warp * MATN * MATM, c_frag, MATN, wmma::mem_row_major);
}
#endif

#if TC == 0
#ifdef TIMING
__global__ void repeat_cc(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c, int n,
                          uint64_t *startclk, uint64_t *stopclk, uint8_t *colmap) {
    uint64_t start, end;
    uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
#else
__global__ void repeat_cc(MATTYPE_AB *a, MATTYPE_AB *b, MATTYPE_C *c, int n) {
#endif
    extern __shared__ MATTYPE_C _c[];
    uint32_t idx = threadIdx.x;
#if CONFIG == 17
    MATTYPE_C _c0 = __float2half2_rn(0.0f);
    MATTYPE_C _c1 = __float2half2_rn(0.0f);
    MATTYPE_C _c2 = __float2half2_rn(0.0f);
    MATTYPE_C _c3 = __float2half2_rn(0.0f);
    MATTYPE_C _c4 = __float2half2_rn(0.0f);
    MATTYPE_C _c5 = __float2half2_rn(0.0f);
    MATTYPE_C _c6 = __float2half2_rn(0.0f);
    MATTYPE_C _c7 = __float2half2_rn(0.0f);
    MATTYPE_C j = __float2half2_rn(1.0f);
    MATTYPE_C k = __float2half2_rn(1.0f);
#else
    MATTYPE_C _c0 = 0;
    MATTYPE_C _c1 = 0;
    MATTYPE_C _c2 = 0;
    MATTYPE_C _c3 = 0;
    MATTYPE_C _c4 = 0;
    MATTYPE_C _c5 = 0;
    MATTYPE_C _c6 = 0;
    MATTYPE_C _c7 = 0;
    MATTYPE_C j = 1;
    MATTYPE_C k = 1;
#endif
#ifdef TIMING
    start = clock64();
#endif
    for (uint32_t i = 0; i < n; i++) {
        asm volatile ("{\t\n"
        INSTRUCTION
        "}" : "+" XSTR(OP_TYPE) (FUNC(_c0)),
              "+" XSTR(OP_TYPE) (FUNC(_c1)),
              "+" XSTR(OP_TYPE) (FUNC(_c2)),
              "+" XSTR(OP_TYPE) (FUNC(_c3)),
              "+" XSTR(OP_TYPE) (FUNC(_c4)),
              "+" XSTR(OP_TYPE) (FUNC(_c5)),
              "+" XSTR(OP_TYPE) (FUNC(_c6)),
              "+" XSTR(OP_TYPE) (FUNC(_c7)) :
        XSTR(OP_TYPE) (FUNC(j)), XSTR(OP_TYPE) (FUNC(k))
    );
    }

    _c[idx] = _c0 + _c1 + _c2 + _c3 + _c4 + _c5 + _c6 + _c7;
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

double run_repeat(size_t warps, size_t blocks, size_t n, double *flops,
                  uint64_t *cycles, double *cycle_flops, int exp) {
    MATTYPE_AB *dev_a, *dev_b;
    MATTYPE_C *c, *dev_c;
    uint64_t *startclk, *dev_startclk, *stopclk, *dev_stopclk;
    uint8_t *colmap, *dev_colmap;
    size_t threads = warps * WARP_SIZE;
    size_t tot_warps = warps * blocks;
    struct timespec start, end;
    uint32_t matsize_A = MATM * MATK;
    uint32_t matsize_B = MATN * MATK;
    uint32_t matsize_C = MATN * MATM;
    checkCudaCall(cudaMalloc((void**) &dev_a, tot_warps * matsize_A * sizeof(MATTYPE_AB)));
    checkCudaCall(cudaMalloc((void**) &dev_b, tot_warps * matsize_B * sizeof(MATTYPE_AB)));
    checkCudaCall(cudaMalloc((void**) &dev_c, tot_warps * matsize_C * sizeof(MATTYPE_C)));
    checkCudaCall(cudaMalloc((void**) &dev_startclk, tot_warps * sizeof(uint64_t)));
    checkCudaCall(cudaMalloc((void**) &dev_stopclk, tot_warps * sizeof(uint64_t)));
    checkCudaCall(cudaMalloc((void**) &dev_colmap, tot_warps * sizeof(uint8_t)));

    c = (MATTYPE_C*) malloc(tot_warps * matsize_C * sizeof(MATTYPE_C));
    startclk = (uint64_t*) malloc(tot_warps * sizeof(uint64_t));
    stopclk = (uint64_t*) malloc(tot_warps * sizeof(uint64_t));
    colmap = (uint8_t*) malloc(tot_warps * sizeof(uint8_t));

    for (size_t w = 0; w < tot_warps; w++) {
        init_mat<<<1, 256>>>(dev_a + w * matsize_A, 256, matsize_A);
        init_mat<<<1, 256>>>(dev_b + w * matsize_B, 256, matsize_B);
    }
    checkCudaCall(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &start);

    #if TC == 1
    #ifdef TIMING
    repeat_tc<<<blocks, threads>>>(dev_a, dev_b, dev_c, n, dev_startclk, dev_stopclk, dev_colmap);
    #else
    repeat_tc<<<blocks, threads>>>(dev_a, dev_b, dev_c, n);
    #endif
    #else
    #ifdef TIMING
    repeat_cc<<<blocks, threads, threads * sizeof(MATTYPE_C)>>>(dev_a, dev_b, dev_c, n, dev_startclk, dev_stopclk, dev_colmap);
    #else
    repeat_cc<<<blocks, threads, threads * sizeof(MATTYPE_C)>>>(dev_a, dev_b, dev_c, n);
    #endif
    #endif
    checkCudaCall(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);

    checkCudaCall(cudaMemcpy(c, dev_c, tot_warps * matsize_C * sizeof(MATTYPE_C), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(startclk, dev_startclk, tot_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(stopclk, dev_stopclk, tot_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(colmap, dev_colmap, tot_warps * sizeof(uint8_t), cudaMemcpyDeviceToHost));

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
        sprintf(fname, "%s/%02d_%d_%05d_%02d.csv", dir, CONFIG, n, blocks, warps);
        FILE *f = fopen(fname, "w");
        for (uint32_t i = 0; i < tot_warps; i++) {
            if (colmap[i]) {
                fprintf(f, "%d %lu %lu\n", i, startclk[i] - min, stopclk[i] - min);
            }
        }

        fclose(f);
    }
#endif
    size_t ops = n * MATN * MATM * MATK * 2 * tot_warps;

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
    checkCudaCall(cudaFree(dev_a));
    checkCudaCall(cudaFree(dev_b));
    checkCudaCall(cudaFree(dev_c));
    checkCudaCall(cudaFree(dev_startclk));
    checkCudaCall(cudaFree(dev_stopclk));
    checkCudaCall(cudaFree(dev_colmap));
    return t;
}

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    checkCudaCall(cudaGetDeviceProperties(&prop, 0));
    int ch, n = 1;
    int min_warps = -1;
    int max_warps = 32;
    int min_blocks = -1;
    int max_blocks = 1;
    uint64_t cycles;
    uint32_t mode = 0;
    uint32_t exp = 0;
    double flops, cycle_flops;

    while ((ch = getopt(argc, argv, "d:n:w:W:b:B:m:e:")) != -1)
    {
        switch(ch) {
            case 'd': dir = optarg; break;
            case 'm': mode = strtol(optarg, 0, 10); break;
            case 'e': exp = strtol(optarg, 0, 10); break;
            case 'n': n = strtol(optarg, 0, 10); break;
            case 'w': min_warps = strtol(optarg, 0, 10); break;
            case 'W': max_warps = strtol(optarg, 0, 10); break;
            case 'b': min_blocks = strtol(optarg, 0, 10); break;
            case 'B': max_blocks = strtol(optarg, 0, 10); break;
        }
    }

    if (min_warps == -1) {
        min_warps = max_warps;
    }

    if (min_blocks == -1) {
        min_blocks = max_blocks;
    }

    for (uint32_t blocks = min_blocks; blocks <= max_blocks; blocks++) {
        for (uint32_t warps = min_warps; warps <= max_warps; warps++) {
            double t;
            if (mode == 0) {
                t = run_repeat(warps, blocks, n, &flops, &cycles, &cycle_flops, exp);
            } else {
                t = run_repeat(warps, blocks * prop.multiProcessorCount, n, &flops, &cycles, &cycle_flops, exp);
            }
            printf("%d %d %d %.6e %.6e %lu %.6e\n", n, warps, blocks, t, flops, cycles, cycle_flops);
        }
    }

}
