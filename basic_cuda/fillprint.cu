#include <stdio.h>
#include <stddef.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "fillprint.h"


__global__ void fillprintCUDAbuf(int* buf, size_t size)
{
  for (int i=0; i<size; ++i)
    {
      buf[i] = i;
      printf("%d ", buf[i]);
    }
  printf("\n");
}



void fillprintCUDAbuf_wrap(int* buf, size_t size)
{
  fillprintCUDAbuf<<<1, 1>>>(buf, size);
}
