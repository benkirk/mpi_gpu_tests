#!/bin/bash
#PBS -A <project_code>
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l select=2:ncpus=4:mpiprocs=4:ompthreads=1:ngpus=4

### Set temp to scratch
export TMPDIR=/glade/gust/scratch/${USER}/tmp && mkdir -p $TMPDIR

. config_env.sh || exit 1

### Interrogate Environment
env | egrep "PBS|MPI|THREADS|PALS" | sort | uniq

TESTS_DIR=${inst_dir}

[ -d ${TESTS_DIR} ] || { echo "cannot find tests: ${TESTS_DIR}"; exit 1; }

# Enable GPU support in the MPI library
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

for tool in $(find ${TESTS_DIR} -type f -executable -name "osu_alltoall*" | sort); do

    echo && echo && echo "********* Inter-Node-CPU *****************"
    echo ${tool} && mpiexec -n 8 --ppn 4 ./get_local_rank ${tool}

    # this should be the problem config
    echo && echo && echo "********* Inter-Node-GPU *****************"
    echo ${tool} && mpiexec -n 8 --ppn 4 ./get_local_rank ${tool} -d managed
done

echo "Done at $(date)"
