#include <getopt.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

#ifndef LOAD_POLICY
#define LOAD_POLICY .ca
#endif

#ifndef STORE_POLICY
#define STORE_POLICY .cg
#endif

#define STR(x) #x
#define XSTR(s) STR(s)

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

static void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr,"checkCuda: %s %s %d\n", cudaGetErrorString(result), file, line);
        exit(1);
    }
}

#define checkCudaCall(res) { checkCuda((res), __FILE__, __LINE__); }

typedef struct {
    uint32_t cols;
    uint32_t rows;
    uint32_t tpb;
    uint32_t blocks;
    uint32_t repeat;
    uint32_t method;
} args_t;

__global__ void init_mat(FTYPE *A, uint32_t num_threads, uint32_t lim) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < lim; i += num_threads) {
        A[i] = i;
    }
}

__global__ void check_mat(FTYPE *B, uint32_t num_threads, uint32_t lim,
                          uint32_t rows, uint32_t cols) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < lim; i += num_threads) {
        uint32_t row = i / cols;
        uint32_t col = i % cols;
        if (B[col * rows + row] != i) {
            printf("index %d (%d) incorrect! Should be %d is %d\n", i, col * rows + row, i, (uint32_t) B[col * rows + row]);
        }
    }
}

__global__ void transpose(FTYPE *A, FTYPE *B, uint32_t num_threads,
                          uint32_t lim, uint32_t rows, uint32_t cols) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < lim; i += num_threads) {
        uint32_t row = i / cols;
        uint32_t col = i % cols;
        FTYPE *in = A + i;
        FTYPE *out = B + col * rows + row;

        asm volatile ("{\t\n"
                ".reg .f" XSTR(BIT_SIZE) " data;\n\t"
                "ld.global" XSTR(LOAD_POLICY) ".f" XSTR(BIT_SIZE) " data, [%0];\n\t"
                "st.global" XSTR(STORE_POLICY) ".f" XSTR(BIT_SIZE) " [%1], data;\n\t"
                "}" :: "l"(in) "l" (out) : "memory"
        );
    }
}

__global__ void transpose2(FTYPE *A, FTYPE *B, uint32_t num_threads,
                          uint32_t lim, uint32_t rows, uint32_t cols) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < lim; i += num_threads) {
        uint32_t row = i / cols;
        uint32_t col = i % cols;
        FTYPE *out = B + i;
        FTYPE *in = A + col * rows + row;

        asm volatile ("{\t\n"
                ".reg .f" XSTR(BIT_SIZE) " data;\n\t"
                "ld.global" XSTR(LOAD_POLICY) ".f" XSTR(BIT_SIZE) " data, [%0];\n\t"
                "st.global" XSTR(STORE_POLICY) ".f" XSTR(BIT_SIZE) " [%1], data;\n\t"
                "}" :: "l"(in) "l" (out) : "memory"
        );
    }
}

__global__ void transpose_blocks(FTYPE *A, FTYPE *B, uint32_t num_threads,
                                 uint32_t lim, uint32_t rows, uint32_t cols) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < lim; i += num_threads) {
        uint32_t col = (i / rows) * 8;
        uint32_t row = i % rows;
        for (uint32_t j = 0; j < 8 && col < cols; j++, col++) {
            FTYPE *in = A + row * cols + col;
            FTYPE *out = B + col * rows + row;
            asm volatile ("{\t\n"
                    ".reg .f" XSTR(BIT_SIZE) " data;\n\t"
                    "ld.global" XSTR(LOAD_POLICY) ".f" XSTR(BIT_SIZE) " data, [%0];\n\t"
                    "st.global" XSTR(STORE_POLICY) ".f" XSTR(BIT_SIZE) " [%1], data;\n\t"
                    "}" :: "l"(in) "l" (out) : "memory"
            );
        }
    }
}

__global__ void transfer(FTYPE *A, FTYPE *B, uint32_t num_threads,
                         uint32_t lim, uint32_t rows, uint32_t cols) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = idx; i < lim; i += num_threads) {
        uint32_t row = i / cols;
        uint32_t col = i % cols;
        FTYPE *out = B + row * cols + col;
        FTYPE *in = A + i;

        asm volatile ("{\t\n"
                ".reg .f" XSTR(BIT_SIZE) " data;\n\t"
                "ld.global" XSTR(LOAD_POLICY) ".f" XSTR(BIT_SIZE) " data, [%0];\n\t"
                "st.global" XSTR(STORE_POLICY) ".f" XSTR(BIT_SIZE) " [%1], data;\n\t"
                "}" :: "l"(in) "l" (out) : "memory"
        );
    }
}


double run(args_t *args) {
    FTYPE *dev_a, *dev_b;
    struct timespec start, end;
    uint32_t lim = args->cols * args->rows;
    checkCudaCall(cudaMalloc((void**) &dev_a, args->cols * args->rows * sizeof(FTYPE)));
    checkCudaCall(cudaMalloc((void**) &dev_b, args->cols * args->rows * sizeof(FTYPE)));
    init_mat<<<args->blocks, args->tpb>>>(dev_a, args->blocks * args->tpb, lim);
    checkCudaCall(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &start);
    if (args->method == 0) {
        transfer<<<args->blocks, args->tpb>>>(dev_a, dev_b, args->blocks * args->tpb, lim, args->rows, args->cols);
    } else if (args->method == 1) {
        transpose<<<args->blocks, args->tpb>>>(dev_a, dev_b, args->blocks * args->tpb, lim, args->rows, args->cols);
    } else if (args->method == 2) {
        transpose2<<<args->blocks, args->tpb>>>(dev_a, dev_b, args->blocks * args->tpb, lim, args->rows, args->cols);
    } else {
        transpose_blocks<<<args->blocks, args->tpb>>>(dev_a, dev_b, args->blocks * args->tpb, lim, args->rows, args->cols);
    }
    checkCudaCall(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);

    // check_mat<<<args->blocks, args->tpb>>>(dev_b, args->blocks * args->tpb, lim, args->rows, args->cols);
    checkCudaCall(cudaFree(dev_a));
    checkCudaCall(cudaFree(dev_b));

    return (end.tv_sec + (end.tv_nsec / 1000000000.0)) -
           (start.tv_sec + (start.tv_nsec / 1000000000.0));;
}

int main(int argc, char *argv[]) {
    int ch;
    args_t args;
    args.cols = 1000;
    args.rows = 1000;
    args.tpb = 1024;
    args.blocks = 28;
    args.repeat = 10;
    args.method = 1;

    while ((ch = getopt(argc, argv, "r:n:m:t:b:v:")) != -1)
    {
        switch(ch) {
        case 'n': args.cols = strtol(optarg, 0, 10); break;
        case 'm': args.rows = strtol(optarg, 0, 10); break;
        case 't': args.tpb = strtol(optarg, 0, 10); break;
        case 'b': args.blocks = strtol(optarg, 0, 10); break;
        case 'r': args.repeat = strtol(optarg, 0, 10); break;
        case 'v': args.method = strtol(optarg, 0, 10); break;
        }
    }

    double t[args.repeat];
    double t_sum = 0, t_avg, t_std;
    for (uint32_t r = 0; r < args.repeat; r++) {
        t[r] = run(&args);
        t_sum += t[r];
    }
    t_avg = t_sum / args.repeat;
    t_sum = 0;
    for (uint32_t r = 0; r < args.repeat; r++) {
        t_sum += (t[r] - t_avg) * (t[r] - t_avg);
    }
    t_std = sqrt(t_sum / args.repeat);
    printf("%d %d %d %d %.6e %.6e %.6e\n", args.cols, args.rows, args.tpb,
           args.blocks, t_avg, t_std,
           args.cols * args.rows * sizeof(FTYPE) / t_avg);
}