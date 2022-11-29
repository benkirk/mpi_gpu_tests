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


// anonymous namespace to hold "global" variables, but restricted to this translation unit
namespace {
  int rank, nranks, local_rank;
  char hn[256];
}



// https://www.open-mpi.org/faq/?category=runcuda
void select_cuda_device()
{
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



int* allocate(const std::size_t cnt, const MemType mem_type)
{
  int* buffer = NULL;

  switch (mem_type)
    {
    case CPU:
      buffer = (int *) malloc(cnt*sizeof(int));
      assert (NULL != buffer);
      std::cout << "Rank " << std::setw(3) << rank
                << " successfully allocated "
		<< cnt*sizeof(int)
		<< " bytes of CPU memory\n";
      for (int i=0; i<cnt; ++i)
        buffer[i] = i;
      // for (int i=0; i<std::min(cnt,static_cast<std::size_t>(10)); ++i) {
      //   if (i < 9) std::cout << buffer[i] << " ";
      //   else       std::cout << "...\n";
      // }
      return buffer;

#ifdef HAVE_CUDA
    case GPU_Device:
      CUDA_CHECK(cudaMalloc((void**) &buffer, cnt*sizeof(int)));
      std::cout << "Rank " << std::setw(3) << rank
                << " successfully allocated "
		<< cnt*sizeof(int)
		<< " bytes of GPU memory\n";
      fillprintCUDAbuf (buffer, cnt);
      return buffer;

    case GPU_Managed:
      CUDA_CHECK(cudaMallocManaged((void**) &buffer, cnt*sizeof(int)));
      std::cout << "Rank " << std::setw(3) << rank
                << " successfully allocated "
		<< cnt*sizeof(int)
		<< " bytes of managed GPU memory\n";
      fillprintCUDAbuf (buffer, cnt);
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



int* copy_dev_to_host (const std::size_t cnt, int *dev, int *host)
{
#ifdef HAVE_CUDA
  CUDA_CHECK(cudaMemcpy(host, dev, sizeof(int)*cnt, cudaMemcpyDeviceToHost));
#else
  assert(false);
#endif

  return host;
}



int* copy_host_to_dev (const std::size_t cnt, int *host, int *dev)
{
#ifdef HAVE_CUDA
  CUDA_CHECK(cudaMemcpy(dev, host, sizeof(int)*cnt, cudaMemcpyHostToDevice));
#else
  assert(false);
#endif

  return dev;
}



int main (int argc, char **argv)
{
  int opt;
  const std::size_t bufcnt_MAX = 1e9;
  int *buf = NULL;
  int *hbuf = NULL;
  MemType mem_type = CPU;
  bool copy_to_host = false;

  while((opt = getopt(argc, argv, ":dmc")) != -1)
    {
      switch(opt)
        {
	case 'd':
	  mem_type = GPU_Device;
	  break;

	case 'm':
	  mem_type = GPU_Managed;
	  break;

	case 'c':
	  copy_to_host = true;
	  break;

	case '?':
	  printf("unknown option: %c\n", opt);
	  break;
        }
    }

  gethostname(hn, sizeof(hn) / sizeof(char));

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
            << " / " << std::string (hn)
	    << ", running " << argv[0] << " on "
	    << std::setw(3) << nranks << " rank(s)"
	    << std::endl;


  select_cuda_device();

  //--------------------
  buf = allocate(bufcnt_MAX, mem_type);
  if (copy_to_host) hbuf = allocate(bufcnt_MAX, CPU);



  std::vector<std::size_t> cnts{1};
  while (cnts.back() < bufcnt_MAX)
    cnts.push_back(10*cnts.back());

  MPI_Barrier(MPI_COMM_WORLD);

  for (auto & bufcnt : cnts)
    {
      if (bufcnt >= bufcnt_MAX) break;

      const double t_start = MPI_Wtime();

      if (rank%2 == 0) // even ranks send...
        {
          // send directy from'buf, wherever it lies...
          if (!copy_to_host)
            MPI_Send (buf, bufcnt, MPI_INT, /* dest = */ (rank + 1) % nranks, /* tag = */ 100, MPI_COMM_WORLD);

          else
            {
              if (mem_type != CPU) copy_dev_to_host (bufcnt, buf, hbuf);
              MPI_Send (hbuf, bufcnt, MPI_INT, /* dest = */ (rank + 1) % nranks, /* tag = */ 100, MPI_COMM_WORLD);
            }
        }

      else // odd ranks recv
        {
          // recv directy from'buf' ; wherever it lies...
          if (!copy_to_host)
            MPI_Recv (buf, bufcnt, MPI_INT, /* src = */ (rank - 1), /* tag = */ 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          else
            {
              MPI_Recv (hbuf, bufcnt, MPI_INT, /* src = */ (rank - 1), /* tag = */ 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              if (mem_type != CPU) copy_host_to_dev (bufcnt, hbuf, buf);
            }
        }

      const double elapsed = MPI_Wtime() - t_start;
      if (rank == 0)
        std::cout << std::setw(10) << bufcnt << " : "
                  << elapsed << " (sec)"
                  << std::endl;
    }

  deallocate(buf,  mem_type);
  deallocate(hbuf, CPU);

  MPI_Finalize();

  return 0;
}
