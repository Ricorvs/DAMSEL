# Memory throughput

This microbenchmark is used for section 7.4 and consists of code for matrix
transposition. As before the execution can be influenced with compile time
arguments and runtime arguments.

## Requirements

This microbenchmark requires Compute Capability 2.0 or higher, which
corresponds to the Fermi architecture or later.

## NVCC flags

The following flags allow the user to control certain properties of the
executable at compile time when compiling the microbenchmarks manually (which
is to say _not_ via `run_suite.py`). The simplest way to modify these flags is
to append them to the `NVCCFLAGS` environment variable. For example, to set the
load policy to `.ca`, one may append `-DLOAD_POLICY=.ca` to this environment
variable.

| Compile flag | Values | Effect
|---|---|---|
|`LOAD_POLICY` | `.ca`, `.cg`, `.cs`, `.lu`, `.cv` | Set the cache policy used for loads |
|`STORE_POLICY` | `.wb`, `.cg`, `.cs`, `.wt` | Set the cache policy used for stores |
|`BIT_SIZE` | `32`, `64` | Set the element size used in transpose |

## Runtime flags

The following run-time parameters are available when running the `transpose`
executable manually (again, _not_ via `run_suite.py`).

| Runtime arguments | Values | Effect
|---|---|---|
|`-n` | `int` | Set the number of columns |
|`-m` | `int` | Set the number of rows |
|`-t` | `int [1, 1024]` | Set the number of threads per block |
|`-b` | `int` | Set the number of blocks |
|`-r` | `int` | Set the number of repetitions |
|`-v` | `int [0, 3]` | Set the used function <br> `0`: Linear copy<br> `1`: Row major reads<br> `2`: Column major reads<br>`3`: Cache optimised  |

## Experimental suite

The experimental suite provided for this microbenchmark performs three distinct
experiments:

1. A broad exploration of the parameter space, in which the number of blocks,
   the block size, and the matrix size are variables.
2. An exploration of the effect of matrix size, in which the block size and
   block count are fixed, but the matrix size is variable.
3. An exploration of the effect of block count effects, in which the block size
   and matrix size are fixed, but the number of blocks is variable.

For each of these sections, configuration parameters are given in the code,
which are marked with a `CONFIG` marker. The configuration parameters are as
follows:

- `blocks`: the number of blocks to use in the grid
- `tbp`: the number of threads per block
- `size`: the sizes (in tuples) of the matrices
- `repeats`: the number of times each experiment is repeated; this parameter
  must have length equal to the length of `size`, as the _i_-th value of `size`
  is repeated a number of times equal to the _i_-th value of `repeats

Please note that this entire experimental suite is repeated twenty times, once
for every possible combination of the five load policies and the four store
policies.

## Output

The output of individual runs of `transpose` is a space-separated line of seven
values with the following meanings (in order):

1. Number of columns in the matrices
2. Number of rows in the matrices
3. Number of threads per block
4. Number of blocks
5. Mean execution time
6. Standard deviation of execution time
7. Mean execution time per matrix element

Please note that the first four values are simply repetitions of what is
provided as run-time arguments, but they are provided to facilitate processing
of the data.

If the experimental suite is run, the results are written to a directory called
`output` (this can be changed by supplying a directory name to the Python
script). This file contains three directories, `blocks`, `default`, and
`mat_size`, corresponding to the three experiments described above. Each of
these directories then contains directories formatted as
`[load_policy]_[store_policy]`. This directory, finally, contains
space-separated files named `res_[value].csv`, where `value` takes the same
values as the `-v` flag on the executable, such that `0` indicates a linear
copy, `1` indicates reading from the input matrix in row-major order, `2`
indicates reading in column-major order, and `3` indicates reading in a
cache-optimized order.

For example, the result file `output/default/ca_cg/res_0.csv` contains the
results for our broad parameter exploration using the `.ca` read policy and the
`.cg` write policy, while copying data in a linear fashion.

**Important note:** The resulting files have the `.csv` extension, but they are not technically _comma_-separated files. Rather, they are space-separated.

## Plotting

There is a single gnuplot plotting script available, `plot_combined.gnuplot`. This script requires two arguments, `folder` and `outputfile` passed in as follows:
`gnuplot -e "folder='results/1080ti'; outputfile='1080ti.pdf'" plot_combined.gnuplot`

The following plots are created:
| Filename | Content |
|---|---|
| `blocks_*` | Throughput compared to number of blocks started for all implementations using `.ca` and `.cg` load policies combined with the `.wb` write policy (Figure 4). |
| `matrix_size_*` | Throughput compared to size of the matrix for all implementations using `.ca` and `.cg` load policies combined with the `.wb` write policy. |