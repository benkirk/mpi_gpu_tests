#!/bin/bash
#PBS -A <project_code>
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l select=2:ncpus=2:mpiprocs=2:ompthreads=1:ngpus=2

### Set temp to scratch
[ -d /glade/gust/scratch/${USER} ] && export TMPDIR=/glade/gust/scratch/${USER}/tmp && mkdir -p $TMPDIR

. config_env.sh || exit 1

# Enable GPU support in the MPI library
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export LD_PRELOAD=/usr/lib64/libvma.so.9.1.2
#export LD_PRELOAD=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.6/lib64/libcudart.so.11.0

### Interrogate Environment
env | sort | uniq | egrep -v "_LM|_ModuleTable"

TESTS_DIR=${inst_dir}

[ -d ${TESTS_DIR} ] || { echo "cannot find tests: ${TESTS_DIR}"; exit 1; }

for tool in $(find ${TESTS_DIR} -type f -executable -name osu_bw -o -name osu_bibw -o -name osu_latency | sort); do

    echo ${tool} && ldd ${tool}

    echo && echo && echo "********* Intra-Node-CPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 2 ./get_local_rank ${tool}

    echo && echo && echo "********* Inter-Node-CPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 1 ./get_local_rank ${tool}

    echo && echo && echo "********* Intra-Node-GPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 2 ./get_local_rank ${tool} D D

    # this should be the problem config
    echo && echo && echo "********* Inter-Node-GPU *****************"
    echo ${tool} && mpiexec -n 2 --ppn 1 ./get_local_rank ${tool} D D
done

echo "Done at $(date)"
