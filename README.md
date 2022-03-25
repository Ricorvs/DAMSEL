# Artifact instructions

This README describes what is included in the artifact for the paper "Isolating
GPU Architectural Features using Parallelism-Aware Microbenchmarks". This is
split into 3 separate microbenchmarks:

1. Cache latency and bandwidth (`1.cache`)
2. Memory throughput (`2.transpose`)
3. Arithmetic performance (`3.arithmetic`)

For each of these microbenchmarks, we include the following:

1. C++ and CUDA source code
2. Build system using `make`
3. Python helper scripts for data collection
4. Gnuplot scripts for visualising the data
5. The verbatim results used in the paper itself

## Requirements

In order to compile and run the microbenchmarks in this artifact, the following
must be available:

- GNU Make
    - Other Make-like tools may also work, but we only support GNU Make at this
      time
- CUDA-capable GPU
- CUDA toolkit, version 10 or greater
    - `nvcc` must be available
    - A compatible host-compiler must also be installed for the `nvcc` compiler
      driver to function
- _(Optional)_ Python 3 to run the helper scripts
- _(Optional)_ Gnuplot to reproduce the plots in the paper

It must be noted that, due to the nature of our microbenchmarks, individual
microbenchmarks may require certain features to be present on the GPU. For
example, our tensor core benchmarks require Compute Capability 7.0 or greater.
A complete list of NVIDIA devices and their Compute Capability is [available on
the NVIDIA developer's website](https://developer.nvidia.com/cuda-gpus).

### Tested devices

The artifact has been tested on the following devices:

- NVIDIA GTX 1080 Ti (_Pascal_)
- NVIDIA GTX 2080 Ti (_Turing_)
- NVIDIA A100 (_Ampere_)
- NVIDIA GTX Titan X (_Maxwell_) (part 1 only)
- NVIDIA GTX 1660 Ti (_Turing_) (parts 2 and 3 only)
- NVIDIA GTX 2060 (_Turing_) (parts 2 and 3 only)

## Running experiments

The execution for the benchmarks can be influenced with compile time arguments
as well as runtime arguments. An explanation of these flags is included with
each individual microbenchmark. In addition to being able to indivudually
compile and run experiments, we also provide `run_suite.py`, which
automatically explores (part of) the parameter space for a given
microbenchmark. These Python scripts aid the user in running experiments and
aggregating the results of these experiments.

**Important note:** The `run_suite.py` helper scripts encapsulate both
execution and _compilation_ of the experiments. This means that any manually
compiled executables will be **overwritten and lost** when the helper script is
run.

**Important note:** For the time being, the parameter exploration in the
`run_suite.py` helper scripts is hard-coded in the Python files. This means
that users wanting to use these helper scripts need to modify them in order to
change the parameter space exploration. By default, the values are designed to be suitable for the _NVIDIA GeForce 1080 Ti_ GPU.

In summary, our microbenchmarks are designed to be used in one of two different
ways, depending on the user's preference. Firstly, we provide `run_suite.py`
helper scripts which encapsulate the entire compilation-execution-exploration
cycle. Secondly, we provide a more fine-grained approach where the
microbenchmarks can be compiled (and executed) with custom flags in order to
reproduce a single experiment. This allows users to interact with the binaries
with other programming languages and automated experimental setups, such as
bash.

For precise instructions on building and running the microbenchmarks, please
see the individual `README.md` files included in the relevant microbenchmark's
subdirectory.
