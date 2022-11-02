#!/bin/bash
#PBS -N mpi_cuda
#PBS -A <project_code>
#PBS -j oe
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=4:mpiprocs=2:ompthreads=2:ngpus=2

### Set temp to scratch
export TMPDIR=/glade/gust/scratch/${USER}/tmp && mkdir -p $TMPDIR

. config_env.sh || exit 1

gmake clean
gmake
gmake run

echo && echo && echo "Done at $(date)"
