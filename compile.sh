package="osu-micro-benchmarks"
version="6.1"
src_dir=${package}-${version}
tarball=${src_dir}.tar.gz

[ -f ${tarball} ] || wget https://mvapich.cse.ohio-state.edu/download/mvapich/${tarball} || exit 1
[ -d ${src_dir} ] || tar zxf ${tarball} || exit 1

rm -rf ./modules && mkdir ./modules && cp -r /glade/u/apps/gust/modules/22.08b/Core/cuda ./modules
module use ./modules
module purge
module load crayenv
module avail
module load PrgEnv-gnu craype-accel-nvidia80 libfabric cuda
module list


inst_dir=$(pwd)/install

cd ${src_dir} && rm -rf BUILD && mkdir BUILD && cd BUILD || exit 1
CXX=CC CC=cc FC=ftn F77=${FC} \
       CPPFLAGS="-I${NCAR_ROOT_CUDA}/include" \
       LDFLAGS="-L${NCAR_ROOT_CUDA}/lib64 -lcudart -Wl,-rpath,${NCAR_ROOT_CUDA}/lib64" \
       ../configure --enable-cuda --prefix=${inst_dir} \
    || exit 1

make -j 8 || exit 1
rm -rf ${inst_dir} && make install || exit 1

echo "Done at $(date)"
