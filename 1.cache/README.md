## Cache latency and bandwidth

This microbenchmark is used for sections 6 and 7. It consists of a latency
benchmark as well as a bandwidth benchmark. Using compiler flags the following
execution characteristics can be changed:

| Compile flag | Values | Effect
|---|---|---|
|`CACHE_POLICY` | `.ca`, `.cg`, `.cs`, `.lu`, `.cv` | Set the cache policy used |
|`INSTR` | `0` (`ADD`), `1` (`NEG`), `2` (`MOV`), `3` (`NONE`) | Set the instruction used in bandwidth measurement |
|`BIT_SIZE` | `32`, `64` | Set the element size used in bandwidth measurement |

The execution can be further influenced with runtime arguments, which can be viewed by calling the program with `-h`.

## Experimental Suite
The experimental suite provided for this microbenchmark performs two distinct experiments for each of the possible cache policies:

1. Latency measurements using various methods, regular pchase, texture memory pchase, contention on a single SM, contention on multiple SMs.
2. Bandwidth measurements using various numbers of blocks with varying block sizes. Additionally, the four different instructions are tested using both 32 and 64 bit modes.

## Output
If the experimental suite is run, the results are written to a directory called `output` (this can be changed by supplying a directory name to the Python script). This directory contains a subdirectory for each of the tested cache policies. For each of the cache policies the underlying hierarchy is the same with 5 directories, `default`, `texture`, `single_sm`, `multi_sm`, and `bw`.

The `default` and `texture` folders contain space-separated files named `preheat[0/1]_stride[warps].csv` based on whether preheat is turned on and the selected stride. These files contain six columns: `buffer size`, `minimum latency`, `maximum latency`, `average latency`, `cache hits`, `cache misses`.

The `single_sm` and `multi_sm` folders contain space-separated files named `[value]_warps.csv` where value is the number of simultaneous warps. The first two columns are `#access` and `offset`, then for each warp executed there are two columns, `start` and `end` timestamps for each access.

Finally the `bw` contains a folder hierarchy of `[Bitwidth]/[Instruction]`, each of these folders contain space-separated files named `bw_[blocks]b_[warps].csv` and a file named `bw.csv`. The file `bw.csv` contains a line for each separate execution and has the following columns, `buffer size`, `blocks`, `warps`, `warps active`, `cycles taken`, `bandwidth (bytes/cycle)`. The remaining files contain a line per warp and 6 columns, `SM ID`, `Block ID`, `Warp ID`, `Cycles`, `Bandwidth (bytes/cycle)`.

## Plotting

There are two gnuplot plotting scripts available, `plot_latency.gnuplot` for the latency results and `plot_bw.gnuplot`. Both of these scripts require two arguments, `folder` and `outputfile` passed in as follows:
`gnuplot -e "folder='results/1080ti'; outputfile='1080ti.pdf'" plot_latency.gnuplot`

For `plot_latency.gnuplot` the following plots are created:
| Filename | Content |
|---|---|
| `single_run_latency_*` | Access time for a single run using different strides. |
| `single_sm_impact_*` | The results from running multiple warps on an SM, each page shows a different number of warps (Figure 1) |
| `multi_sm_impact_*` | The results from running multiple warps from multiple blocks on an SM, each page shows a different number of blocks (Figure 2) |

For `plot_bw.gnuplot` the following plots are created:
| Filename | Content |
|---|---|
| `bw_diff_32_*` | Throughput results for 32-bit elements, each page contains a different number of warps per block (Figure 3) |
| `bw_diff_64_*` | Throughput results for 32-bit elements, each page contains a different number of warps per block |
| `bw_diff_*` | Throughput results for different bitwidths, each page contains a different number of warps per block |
| `bw_scheduling_*` | Scheduling results for different block/warp configurations (Figure 5) |