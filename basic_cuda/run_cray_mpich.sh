#!/bin/bash

module reset >/dev/null 2>&1
module load cuda >/dev/null 2>&1
module list

# Enable GPU support in the MPI library
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

# default compiler
make --no-print-directory clean
make --no-print-directory
make --no-print-directory run



# gcc
module load gcc/11.2.0 >/dev/null 2>&1
module list

make --no-print-directory clean
make --no-print-directory
make --no-print-directory run



# nvhpc
module load nvhpc >/dev/null 2>&1
module list

make --no-print-directory clean
make --no-print-directory
make --no-print-directory run



# oneapi
module load oneapi >/dev/null 2>&1
module list

make --no-print-directory clean
make --no-print-directory
make --no-print-directory run
