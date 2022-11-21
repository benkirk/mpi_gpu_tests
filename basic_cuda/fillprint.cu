#include <stdio.h>
#include <stddef.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "fillprint.h"


__global__ void fillCUDAbuf(int* buf, size_t size)
{
  for (int i=0; i<size; ++i)
    buf[i] = i;
}



__global__ void printCUDAbuf(int* buf, size_t size)
{
  const int np = (size < 10) ? size : 10;

  for (int i=0; i<np; ++i)
    {
      if (i < 9) printf("%d ", buf[i]);
      else       printf("...\n");
    }
}



void fillprintCUDAbuf(int* buf, size_t size)
{
  fillCUDAbuf<<<1,  1>>>(buf, size);
  printCUDAbuf<<<1, 1>>>(buf, size);

  cudaDeviceSynchronize();
}
