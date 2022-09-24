#!/usr/bin/env bash
#PBS -q main
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=60:mpiprocs=1:ompthreads=60:ngpus=2

# Handle arguments
user_args=( "$@" )

ncar_stack=no
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --ncar-stack)
            ncar_stack=yes
            ;;
        *)
            ;;
    esac

    shift
done

# build using ncarenv
if [[ ${ncar_stack} == "yes" ]]; then
    cat >config_env.sh <<EOF
module reset
module load gcc/11.2.0 cuda
module list
for tool in CC cc ftn gcc mpiexec; do
    which \${tool}
done
export BUILD_CLASS="ncarenv" && echo "CC: \$(which CC)"
EOF

# build using crayenv
else
    cat >config_env.sh <<EOF
module purge
module load crayenv
module load PrgEnv-gnu/8.3.2 craype-x86-rome craype-accel-nvidia80 libfabric cray-pals cpe-cuda
module list
for tool in CC cc ftn gcc mpiexec; do
    which \${tool}
done
export BUILD_CLASS="crayenv" && echo "CC: \$(which CC)"
EOF
fi

cat >>config_env.sh  <<EOF
package="osu-micro-benchmarks"
version="6.1"
src_dir=\${package}-\${version}
tarball=\${src_dir}.tar.gz
inst_dir=$(pwd)/install-\${BUILD_CLASS}
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
rm -rf ${inst_dir} && make install || exit 1


echo && echo && echo "Done at $(date)"
