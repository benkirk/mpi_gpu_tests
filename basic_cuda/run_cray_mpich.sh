#!/usr/bin/env bash
#PBS -q main
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=4:mpiprocs=4:ngpus=4

module reset >/dev/null 2>&1
module load cuda >/dev/null 2>&1
module list

# verbose MPI settings?
export MPICH_ENV_DISPLAY=0

# verbose output during MPI_Init to verify which libfabric provider has been selected?
export MPICH_OFI_VERBOSE=0

# Enable GPU support in the MPI library
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

# use MPI local rank internally to set the CUDA device, not anything from the environment
unset CUDA_VISIBLE_DEVICES

export MPI_ARGS="-n 2 --ppn 1"

# default compiler
make --no-print-directory clean
make --no-print-directory
make --no-print-directory run

#exit 0

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
