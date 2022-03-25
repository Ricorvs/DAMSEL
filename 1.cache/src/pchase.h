#include <cuda.h>
#include <stdio.h>

static void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr,"checkCuda: %s %s %d\n", cudaGetErrorString(result), file, line);
        exit(1);
    }
}

typedef struct pchase_args {
    uint32_t *ret_arr;
    uint32_t ret_arr_size;
    uint64_t arr_size;
    uint32_t stride;
    uint32_t iters;
    uint32_t preheat;
    uint32_t hit_threshold;
    uint32_t repeat;
    void (*body)(pchase_args*, void*);
} pchase_args_t;

typedef struct {
    uint32_t min;
    uint32_t max;
    uint32_t avg;
    uint32_t hits;
    uint32_t miss;
} stats_t;

typedef struct {
    uint32_t blocks;
    uint32_t laneid;
    uint32_t columns;
    uint8_t *colmap;
    uint32_t loadwarp;
    uint32_t heatwarp;
} pchase_sm_args_t;

#define checkCudaCall(res) { checkCuda((res), __FILE__, __LINE__); }

void calculate_stats(uint32_t *values, uint32_t n, uint32_t threshold, stats_t *stats);
void calculate_median(stats_t *stats, uint32_t n, stats_t *out);
int compare_uint(const void *a,const void *b);
void pchase_texture_body(pchase_args_t *args, void *eargs);
void pchase_body(pchase_args_t *args, void *eargs);
void pchase_sm_body(pchase_args_t *args, void *eargs);
void pchase_multi_sm_body(pchase_args_t *args, void *eargs);

void run_pchase(pchase_args_t *args, stats_t *stats, void *eargs);
void export_to_file(uint32_t *arr, uint32_t repeat, uint32_t arr_size,
                    uint32_t cols, uint32_t iters, uint32_t stride,
                    uint8_t *colmap, char *fname);
