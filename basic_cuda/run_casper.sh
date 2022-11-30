#!/usr/bin/env bash
#PBS -q casper
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=2:ngpus=2:mem=40G
#PBS -l gpu_type=v100

module reset >/dev/null 2>&1
module load nvhpc >/dev/null 2>&1
module list

# use MPI local rank internally to set the CUDA device, not anything from the environment
unset CUDA_VISIBLE_DEVICES

unset MPI_ARGS="-n 4"

make --no-print-directory clean
make --no-print-directory
make --no-print-directory run
