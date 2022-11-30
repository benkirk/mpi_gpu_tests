#!/usr/bin/env bash
#PBS -q casper
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=1:mpiprocs=1:ngpus=1:mem=40G
#PBS -l gpu_type=v100

module reset >/dev/null 2>&1
module load nvhpc >/dev/null 2>&1
module list

# use MPI local rank internally to set the CUDA device, not anything from the environment
unset CUDA_VISIBLE_DEVICES

export MPI_ARGS="-n 2"

make --no-print-directory clean
make --no-print-directory
make --no-print-directory run
