#!/usr/bin/env bash

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

rm -rf ./modules && mkdir ./modules && cp -r /glade/u/apps/gust/modules/22.08b/Core/cuda ./modules

cat >config_env.sh  <<EOF
package="osu-micro-benchmarks"
version="6.1"
src_dir=\${package}-\${version}
tarball=\${src_dir}.tar.gz
inst_dir=$(pwd)/install
EOF

# build using ncarenv
if [[ ${ncar_stack} == "yes" ]]; then
    cat >>config_env.sh <<EOF
[ -d ./modules ] && module use ./modules
module reset
module load gcc/11.2.0 cuda
module list
for tool in CC cc ftn gcc mpiexec; do
    which \${tool}
done
export BUILD_CLASS="ncarenv / $(which CC)"
EOF

# build using crayenv
else
    cat >>config_env.sh <<EOF
[ -d ./modules ] && module use ./modules
module purge
module load crayenv
module load PrgEnv-gnu/8.3.2 craype-x86-rome craype-accel-nvidia80 libfabric cray-pals cuda
module list
export CPPFLAGS="-I\${NCAR_ROOT_CUDA}/include"
export LDFLAGS="-L\${NCAR_ROOT_CUDA}/lib64 -lcudart -Wl,-rpath,\${NCAR_ROOT_CUDA}/lib64"
export NVCCFLAGS="-allow-unsupported-compiler"
for tool in CC cc ftn gcc mpiexec; do
    which \${tool}
done
export BUILD_CLASS="crayenv / $(which CC)"
EOF
fi

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
