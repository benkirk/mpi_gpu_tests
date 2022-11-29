#!/usr/bin/env bash
#PBS -q casper
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=4:ngpus=2:mem=10G 
#PBS -l gpu_type=v100

module reset >/dev/null 2>&1
module load nvhpc >/dev/null 2>&1
module list

export MPI_ARGS=""

make --no-print-directory clean
make --no-print-directory
make --no-print-directory run
