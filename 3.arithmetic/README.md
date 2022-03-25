# Arithmetic performance

This microbenchmark is used for sections 8 and 9 of our paper and consists of
code for measuring arithmetic throughput using either CUDA cores or Tensor
cores. The behaviour of this microbenchmark can be influenced using compile
time arguments and runtime arguments:

## Requirements

This microbenchmark requires Compute Capability 6.0 or higher, which
corresponds to the Pascal architecture. The Tensor core benchmarks
(configurations 0 through 16) require require Compute Capability 7.0 or higher,
which corresponds to the Volta microarchitecture or later. Certain data type
configurations using lower-precision formats (configurations 6 through 16)
require Compute Capability 8.0 or higher, which corresponds to the Ampere
architecture.

## NVCC flags

The following flags allow the user to control certain properties of the
executable at compile time when compiling the microbenchmarks manually (which
is to say _not_ via `run_suite.py`). The simplest way to modify these flags is
to append them to the `NVCCFLAGS` environment variable. For example, to disable
FMA, one may append `-DINSTR=1` to this environment variable.

| Compile flag | Values | Effect
|---|---|---|
|`CONFIG` | `int [0, 21]` | Set the configuration to be used (see table below) |
|`TIMING` |  | If this flag is set, fine-grained timing will be enabled |
|`INSTR` | `0, 1` | Set whether to use FMA (0) or non-fused MUL/ADD (1) |

The type of arithmetic operation used can be set using the `CONFIG` compile
flag, these configurations are listed below. Do note that not all
configurations work on every GPU due to lack of support (e.g. no tensor support
on 1080 Ti).

| Config | Core type | Datatype A/B | Datatype C |Size (MxNxK) |
|---|---|---|---|---|
|  `0`| Tensor |`half` | `float`  |`16 x 16 x 16`|
|  `1`| Tensor |`half` | `float`  |`32 x  8 x 16`|
|  `2`| Tensor |`half` | `float`  |` 8 x 32 x 16`|
|  `3`| Tensor |`half` | `half`   |`16 x 16 x 16`|
|  `4`| Tensor |`half` | `half`   |`32 x  8 x 16`|
|  `5`| Tensor |`half` | `half`   |` 8 x 32 x 16`|
|  `6`| Tensor |`uint8` | `int`    |`16 x 16 x 16`|
|  `7`| Tensor |`uint8` | `int`    |`32 x  8 x 16`|
|  `8`| Tensor |`uint8` | `int`    |` 8 x 32 x 16`|
|  `9`| Tensor |`int8` | `int`    |`16 x 16 x 16`|
| `10`| Tensor |`int8` | `int`    |`32 x  8 x 16`|
| `11`| Tensor |`int8` | `int`    |` 8 x 32 x 16`|
| `12`| Tensor |`bf16` | `float`  |`16 x 16 x 16`|
| `13`| Tensor |`bf16` | `float`  |`32 x  8 x 16`|
| `14`| Tensor |`bf16` | `float`  |` 8 x 32 x 16`|
| `15`| Tensor |`tf32` | `float`  |` 8 x 32 x 16`|
| `16`| Tensor |`double` | `double` |` 8 x  8 x  4`|
| `17`| CUDA   |`half` | `float`  |` 4 x  8 x 16`|
| `18`| CUDA   |`half` | `half`   |` 4 x  8 x 16`|
| `19`| CUDA   |`uint8` | `int`    |` 4 x  8 x 16`|
| `20`| CUDA   |`double` | `double` |` 4 x  8 x  4`|
| `21`| CUDA   |`float` | `float`  |` 4 x  8 x 16`|

**Important note:** Configuration 15 is currently not available due to problems
with the TF32 format.

## Runtime flags

The following run-time parameters are available when running both `arithmetic`
and `matmult` executables manually (again, _not_ via `run_suite.py`).

| Runtime arguments | Values | Effect
|---|---|---|
| `-d`| `string` | Set the output folder for the fine grained timing results |
| `-e`| `int` | Set whether fine grained timing results should be exported |
| `-n`| `int` | Set number of iterations used |
| `-w`| `int` | Minimum number of warps per block to use |
| `-W`| `int` | Maximum number of warps per block to use |
| `-b`| `int` | Minimum number of blocks to use |
| `-B`| `int` | Maximum number of blocks to use |
| `-m` | `int` | Multiply number of blocks by the total SM count |

The `matmult` executable takes the following additional arguments:

| Runtime arguments | Values | Effect
|---|---|---|
| `-M`| `int` | Size M of matrices to multiply |
| `-N`| `int` | Size N of matrices to multiply |
| `-K`| `int` | Size K of matrices to multiply |

## Output
If the experimental suite is run, the results are written to a directory called `output` (this can be changed by supplying a directory name to the Python script). This directory contains a subdirectory named `[Instruction/Timing]_[Configuration]` for each of the tested configurations. Each of these directories contains eight files, `blocks.csv`, `kerneltimes[4/8/16/32].csv` and `warps[1_0/1_1/2_1].csv`. All files contain seven columns, `Iterations`, `warps per block`, `blocks`, `elapsed time`, `flops`, `elapsed cycles`, and `ops per cycle`. The last two columns contain only 0 if fine-grained timing is not enabled.  If Timing is enabled an extra directory `kerneltimes` is created which contains files named `[Configuration]_[Iterations]_[Blocks]_[Warps].csv`, each containing three columns, `ID`, `Start`, `End`. This data is used to find the scheduling behaviour.

## Plotting

There are two gnuplot plotting scripts available, `plot_arithmetic_cc.gnuplot` for the CUDA core results and `plot_arithmetic_tc.gnuplot` for the Tensor core results. Both of these scripts require two arguments, `folder` and `outputfile` passed in as follows:
`gnuplot -e "folder='results/1080ti'; outputfile='1080ti.pdf'" plot_arithmetic_cc.gnuplot`

For both scripts the following plots are created:
| Filename | Content |
|---|---|
| `blocks_*` | Throughput compared to blocks started. |
| `blocksdiffs_*` | Throughput compared between different instructions and fine-grained timing turned on or off. |
| `blockstimes_*` | Execution time compared to blocks started. |
| `warps_*` | Throughput compared to warps per block for a single block. (Figure 7, 9) |
| `warps1_*` | Throughput compared to warps per block for entire GPU. (Figure 8, 10) |
| `1_[Configuration]/warptimes_[n].pdf` | Execution time per warp and block, `n` warps per block. |
| `1_[Configuration]/warpdurations_[n].pdf` | Scheduling behaviour for different number of blocks, `n` warps per block (Figure 8, 11) |