#!/bin/bash

module reset >/dev/null 2>&1
module load cuda >/dev/null 2>&1
module load openmpi >/dev/null 2>&1
module list

# Enable GPU support in the MPI library
#export MPICH_GPU_SUPPORT_ENABLED=1
#export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

# default compilers
make --no-print-directory clean
make --no-print-directory
make --no-print-directory run

MPI_ARGS="--mca mpi_common_cuda_verbose 100" make --no-print-directory run_GPUd

# gcc
module load gcc/11.2.0 >/dev/null 2>&1
module list

make --no-print-directory clean
make --no-print-directory
make --no-print-directory run

MPI_ARGS="--mca mpi_common_cuda_verbose 100" make --no-print-directory run_GPUd
