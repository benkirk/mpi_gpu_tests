#!/usr/bin/env bash
#PBS -q main
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=60:mpiprocs=1:ompthreads=60:ngpus=2

# Handle arguments
user_args=( "$@" )

ncar_stack=no
custom_stack=no
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --ncar-stack)
            ncar_stack=yes
            ;;
        --custom-stack)
            custom_stack=yes
            ;;
        *)
            ;;
    esac

    shift
done

cat >config_env.sh <<EOF
top_dir=$(pwd)
EOF

# build using ncarenv
if [[ ${ncar_stack} == "yes" ]]; then
    cat >>config_env.sh <<EOF
module reset
module load gcc/11.2.0 cuda
module list
export BUILD_CLASS="ncarenv" && echo "CC: \$(which CC)"
EOF

# build using custom env
elif [[ ${custom_stack} == "yes" ]]; then
    cat >>config_env.sh <<EOF
module load cuda
module load nvhpc/22.7
module load cray-mpich
module load cray-libsci
module load ncarcompilers
module list
export BUILD_CLASS="custom" && echo "CC: \$(which CC)"
EOF

# build using crayenv
else
    cat >>config_env.sh <<EOF
module purge
module load crayenv
module load PrgEnv-gnu/8.3.3 craype-x86-milan craype-accel-nvidia80 libfabric cray-pals cpe-cuda
module list
export BUILD_CLASS="crayenv" && echo "CC: \$(which CC)"
EOF
fi

cat >>config_env.sh  <<EOF
package="osu-micro-benchmarks"
version="6.1"
src_dir=\${package}-\${version}
tarball=\${package}-\${version}.tar.gz
inst_dir=$(pwd)/install-\${BUILD_CLASS}
for tool in CC cc ftn gcc mpiexec; do
    which \${tool}
done

# Enable verbose MPI settings
export MPICH_ENV_DISPLAY=1

# Enable verbose output during MPI_Init to verify which libfabric provider has been selected
export MPICH_OFI_VERBOSE=1

# Enable GPU support in the MPI library
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
EOF

. config_env.sh || exit 1

[ -f ${tarball} ] || wget https://mvapich.cse.ohio-state.edu/download/mvapich/${tarball} || exit 1
[ -d ${src_dir} ] || tar zxf ${tarball} || exit 1

env | sort | uniq | egrep -v "_LM|_ModuleTable"

cd ${src_dir} && rm -rf BUILD && mkdir BUILD && cd BUILD || exit 1

CXX=$(which CC) CC=$(which cc) FC=$(which ftn) F77=${FC} \
   ../configure --enable-cuda --prefix=${inst_dir} \
    || exit 1

make -j 8 || exit 1
rm -rf ${inst_dir} && make install \
    && cp config.log ${inst_dir} \
    && cp ${top_dir}/config_env.sh ${inst_dir} \
    && ln -sf $(which get_local_rank) ${top_dir}/*_GPU.sh ${inst_dir} \
        || exit 1


echo && echo && echo "Done at $(date)"
