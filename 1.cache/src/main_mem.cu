#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <float.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "pchase.h"

#define BUFSIZE 1024
#define WARP_SIZE 32

#ifndef CACHE_POLICY
#define CACHE_POLICY .ca
#endif

#define STR(x) #x
#define XSTR(s) STR(s)

cudaDeviceProp prop;

char *fname = NULL;
uint64_t min_arr_size, max_arr_size;

__global__ void __determine_write_latency(int iterations, uint32_t *latencies) {
    extern __shared__ unsigned int s_index[];
    extern __shared__ unsigned int s_tvalue[];
    clock_t start, end;
    int i;

    for (i = 0; i < iterations; i++) {
        start = clock();
        s_index[i] = i;
        end = clock();
        s_tvalue[i] = end - start;
    }

    for (i = 0; i < iterations; i++) {
        latencies[i] = s_tvalue[i];
    }
}


uint32_t determine_cache_hit_latency(uint32_t *empty_latency) {
    pchase_args_t args;
    args.arr_size = 128;
    args.stride = 1;
    args.iters = 10000;
    args.preheat = 1;
    args.repeat = 1;
    args.ret_arr_size = 0;
    args.body = pchase_body;

    stats_t stats;
    uint32_t latency, empty;
    uint32_t *dev_empty;
    uint32_t *ret_empty = (uint32_t*) malloc(args.iters * sizeof(uint32_t));

    checkCudaCall(cudaMalloc((void**) &dev_empty, args.iters * sizeof(uint32_t)));
    __determine_write_latency<<<1, 1, args.iters * sizeof(uint32_t)>>>(args.iters, dev_empty);
    checkCudaCall(cudaMemcpy(ret_empty, dev_empty, args.iters * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    run_pchase(&args, &stats, NULL);

    qsort(args.ret_arr, args.iters, sizeof(uint32_t), compare_uint);
    qsort(ret_empty, args.iters, sizeof(uint32_t), compare_uint);

    if (args.iters & 1) {
        latency = args.ret_arr[args.iters / 2];
        empty = ret_empty[args.iters / 2];
    } else {
        latency = (args.ret_arr[args.iters / 2 - 1] + args.ret_arr[args.iters / 2]) / 2;
        empty = (ret_empty[args.iters / 2 - 1] + ret_empty[args.iters / 2]) / 2;
    }

    checkCudaCall(cudaFree(dev_empty));
    free(args.ret_arr);

    if (empty_latency) {
        *empty_latency = empty;
    }

    return latency;
}

uint32_t determine_L1_size(uint32_t hit_threshold, uint32_t *cache_size, uint32_t *line_size, uint32_t *sets) {
    int min_arr_size = 128;
    int max_arr_size = 50000;

    double run_avg = 0;
    int arr_size, i, step = 100;

    uint32_t cache_sets = 0;
    stats_t stats;
    pchase_args_t args;
    args.stride = 1;
    args.iters = 10000;
    args.preheat = 1;
    args.hit_threshold = hit_threshold;
    args.repeat = 10;
    args.body = pchase_body;
    args.ret_arr_size = 0;

    /* Find the L1 size. */
    for (i = 1, arr_size = min_arr_size; arr_size <= max_arr_size; i++, arr_size += step) {
        args.arr_size = arr_size;
        run_pchase(&args, &stats, NULL);
        if (i != 1 && stats.avg > 1.02 * run_avg) {
            if (step != 1) {
                arr_size -= step;
                i--;
                step = 1;
                continue;
            }
            break;
        }
        run_avg += (double) (stats.avg - run_avg) / i;
    }
    *cache_size = arr_size - 1;

    run_avg = 0;

    /* Determine cacheline size. */
    for (i = 1; arr_size <= max_arr_size; i++, arr_size++) {
        args.arr_size = arr_size;
        run_pchase(&args, &stats, NULL);
        if (i != 1 && stats.avg > 1.02 * run_avg) {
            break;
        }
        run_avg += (double) (stats.avg - run_avg) / i;
    }
    *line_size = arr_size - *cache_size - 1;

    /* Determine amount of cache sets. */
    uint32_t prev = 0;
    for (arr_size = *cache_size + *line_size; arr_size <= max_arr_size; arr_size += *line_size, cache_sets++) {
        args.arr_size = arr_size;
        run_pchase(&args, &stats, NULL);
        if (*sets != 1 && stats.avg <= 1.02 * prev) {
            break;
        }
        prev = stats.avg;
    }
    *sets = cache_sets;

    return 0;
}

static void usage(const char *pname)
{
    printf("Usage: %s [OPTION]...\n"
           "\n"
           "  -s NUM     Stride to use.\n"
           "  -r NUM     Number of repeats to do.\n"
           "  -p [0|1]   Turn preheat on or off.\n"
           "  -i NUM     Set the number of iterations.\n"
           "  -n NUM     Set the array size (in number of elements).\n"
           "  -b NUM     Set number of blocks to use.\n"
           "  -l NUM     Set which warp should load (-1 for all).\n"
           "  -h NUM     Set which warp should preheat (-1 for all).\n"
           "  -t NUM     Set highest thread in warp that should participate.\n"
           "  -f FILE    File to export to.\n"
           "  -0         Try to determine L1 characteristics.\n"
           "  -1         Run normal Pchase.\n"
           "  -2         Run Pchase with texture cache.\n"
           "  -3         Run Pchase with multiple warps on an SM.\n"
           "  -4         Run Pchase with multiple blocks (1 thread/block).\n"
           "             Execute only on SM 0.\n"
           "  -?         Print this help.\n"
           ,pname);
    exit(0);
}

void parse_args(int argc, char *argv[], pchase_args_t *args, pchase_sm_args_t *eargs) {
    int ch;
    long temp;

    eargs->blocks = 32;
    eargs->loadwarp = ~0;
    eargs->heatwarp = 0;
    eargs->colmap = NULL;
    eargs->laneid = 1;
    eargs->columns = 1;

    args->ret_arr = NULL;
    args->stride = 32;
    args->iters = 192;
    args->ret_arr_size = 0;
    min_arr_size = 0;
    max_arr_size = 0;
    args->preheat = 1;
    args->repeat = 1;
    args->body = pchase_body;

    while ((ch = getopt(argc, argv, "o:s:r:p:i:n:N:b:l:h:t:f:01234?")) != -1)
    {
        switch(ch) {
        case 's': args->stride = strtol(optarg, 0, 10); break;
        case 'r': args->repeat = strtol(optarg, 0, 10); break;
        case 'p': args->preheat = strtol(optarg, 0, 10); break;
        case 'i': args->iters = strtol(optarg, 0, 10); break;
        case 'n': min_arr_size = strtol(optarg, 0, 10); break;
        case 'N': max_arr_size = strtol(optarg, 0, 10); break;
        case 'b': eargs->blocks = strtol(optarg, 0, 10); break;
        case 'l':
            temp = strtol(optarg, 0, 10);
            eargs->loadwarp = (temp == -1) ? ~0 : (uint32_t) temp;
            break;
        case 'h':
            temp = strtol(optarg, 0, 10);
            eargs->heatwarp = (temp == -1) ? ~0 : (uint32_t) temp;
            break;
        case 't': eargs->laneid = strtol(optarg, 0, 10); break;
        case 'f': fname = optarg; break;
        case '0': args->body = NULL; break;
        case '1':
            args->body = pchase_body;
            args->ret_arr_size = 0;
            break;
        case '2':
            args->body = pchase_texture_body;
            args->ret_arr_size = 0;
            break;
        case '3':
            args->body = pchase_sm_body;
            break;
        case '4':
            args->body = pchase_multi_sm_body;
            break;
        case '?': default: usage(argv[0]);
        }

        if (args->body == pchase_multi_sm_body) {
            eargs->blocks *= prop.multiProcessorCount;
        }
        if (args->body == pchase_sm_body || args->body == pchase_multi_sm_body) {
            args->ret_arr_size =  eargs->laneid * eargs->blocks * args->iters * 2;
            eargs->columns = eargs->blocks * 2;
        }
    }

    if (min_arr_size == 0 && max_arr_size == 0) {
        min_arr_size = 6144;
        max_arr_size = 6144;
    } else if (min_arr_size == 0) {
        min_arr_size = max_arr_size;
    } else if (max_arr_size == 0) {
        max_arr_size = min_arr_size;
    }
}

int mkpath(char* file_path, mode_t mode) {
    assert(file_path && *file_path);
    for (char* p = strchr(file_path + 1, '/'); p; p = strchr(p + 1, '/')) {
        *p = '\0';
        if (mkdir(file_path, mode) == -1) {
            if (errno != EEXIST) {
                *p = '/';
                return -1;
            }
        }
        *p = '/';
    }
    return 0;
}


int main(int argc, char *argv[]) {
    pchase_sm_args_t eargs;
    pchase_args_t args;
    stats_t stats;
    uint32_t cache_size, line_size, cache_sets;

    uint32_t latency, empty_latency;
    checkCudaCall(cudaGetDeviceProperties(&prop, 0));

    parse_args(argc, argv, &args, &eargs);

    latency = determine_cache_hit_latency(&empty_latency);
    args.hit_threshold = (uint32_t) (1.5f * latency);

    if (!args.body) {
        determine_L1_size(args.hit_threshold, &cache_size, &line_size, &cache_sets);
        fprintf(stderr, "%d Size, %d line, %d sets\n", cache_size, line_size, cache_sets);
        exit(0);
    }

    for (uint64_t arr_size = min_arr_size; arr_size <= max_arr_size; arr_size += 8) {
        args.arr_size = arr_size;
        run_pchase(&args, &stats, &eargs);
        printf("%d %6u %6u %6u %6u %6u\n", arr_size * sizeof(uint32_t),
               stats.min, stats.max, stats.avg, stats.hits, stats.miss);

        if (min_arr_size != max_arr_size) {
            free(args.ret_arr);
            args.ret_arr = NULL;
            if (eargs.colmap) {
                free(eargs.colmap);
                eargs.colmap = NULL;
            }
        }
    }


    if (fname) {
        mkpath(fname, 0755);
        export_to_file(args.ret_arr, args.repeat, args.arr_size, eargs.columns, args.iters, args.stride, eargs.colmap, fname);
    }

    if (args.ret_arr) {
        free(args.ret_arr);
    }
    if (eargs.colmap) {
        free(eargs.colmap);
        eargs.colmap = NULL;
    }
    return 0;
}
