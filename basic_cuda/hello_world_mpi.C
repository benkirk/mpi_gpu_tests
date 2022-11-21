#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include "mpi.h"
#ifdef HAVE_CUDA
#  include "cuda.h"
#  include "cuda_runtime.h"
#  include "fillprint.h"
#endif // #ifdef HAVE_CUDA

enum MemType { CPU=0, GPU_Device, GPU_Managed };



// https://www.open-mpi.org/faq/?category=runcuda
void select_cuda_device()
{
  int rank=0, local_rank = -1;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
  }
#ifdef HAVE_CUDA
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  cudaSetDevice(local_rank % num_devices);
#endif // #ifdef HAVE_CUDA
}



int* allocate(const std::size_t size, const MemType mem_type)
{
  int* buffer = NULL;

  switch (mem_type)
    {
    case CPU:
      buffer = (int *) malloc(size*sizeof(int));
      assert (NULL != buffer);
      std::cout << "Successfully allocated "
		<< size*sizeof(int)
		<< " bytes of CPU memory\n";
      for (int i=0; i<size; ++i)
        buffer[i] = i;
      for (int i=0; i<std::min(size,static_cast<std::size_t>(10)); ++i) {
        if (i < 9) std::cout << buffer[i] << " ";
        else       std::cout << "...\n";
      }
      return buffer;

#ifdef HAVE_CUDA
    case GPU_Device:
      CUDA_CHECK(cudaMalloc((void**) &buffer, size*sizeof(int)));
      std::cout << "Successfully allocated "
		<< size*sizeof(int)
		<< " bytes of GPU memory\n";
      fillprintCUDAbuf (buffer, size);
      return buffer;

    case GPU_Managed:
      CUDA_CHECK(cudaMallocManaged((void**) &buffer, size*sizeof(int)));
      std::cout << "Successfully allocated "
		<< size*sizeof(int)
		<< " bytes of managed GPU memory\n";
      fillprintCUDAbuf (buffer, size);
      return buffer;
#endif // #ifdef HAVE_CUDA

    default:
      assert (false);
      break;
    }


  return NULL;
}



int deallocate (int *buffer, const MemType mem_type)
{
  if (NULL == buffer)
    return 0;

  switch (mem_type)
    {
    case CPU:
      free(buffer);
      buffer = NULL;
      break;

#ifdef HAVE_CUDA
    case GPU_Device:
    case GPU_Managed:
      CUDA_CHECK(cudaFree(buffer));
      buffer = NULL;
      break;
#endif // #ifdef HAVE_CUDA

    default:
      assert (false);
      break;
    }

  return 0;
}



int* copy_dev_buf_to_host (const std::size_t size, int *dev, int *host)
{
  // TODO
  return host;
}



int main (int argc, char **argv)
{
  int nranks, rank, opt;
  const std::size_t bufsize = 1000;
  int *buf = NULL;
  int *hbuf = NULL;
  MemType mem_type = CPU;

  while((opt = getopt(argc, argv, ":dm")) != -1)
    {
      switch(opt)
        {
	case 'd':
	  mem_type = GPU_Device;
	  break;

	case 'm':
	  mem_type = GPU_Managed;
	  break;

	case '?':
	  printf("unknown option: %c\n", opt);
	  break;
        }
    }

  MPI_Init(&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  if (nranks%2)
    {
      std::cerr << "ERROR: this requires exactly even number of ranks!\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
      assert(false);
    }

  std::cout << "Hello from " << std::setw(3) << rank
	    << ", running " << argv[0] << " on "
	    << std::setw(3) << nranks << " rank(s)"
	    << std::endl;


  select_cuda_device();

  //--------------------
  buf = allocate(bufsize, mem_type);

  if (rank%2 == 0) // even ranks send...
    MPI_Send (buf, bufsize, MPI_INT, /* dest = */ (rank + 1) % nranks, /* tag = */ 100, MPI_COMM_WORLD);

  else // odd ranks recv
    MPI_Recv (buf, bufsize, MPI_INT, /* src = */ (rank - 1), /* tag = */ 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  deallocate(buf, mem_type);

  MPI_Finalize();

  return 0;
}
