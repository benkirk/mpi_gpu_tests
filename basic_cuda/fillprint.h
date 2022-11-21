#ifndef fillprint_h__
#define fillprint_h__

#include <stddef.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

  void fillprintCUDAbuf (int* buf, size_t);

  inline
  int CUDA_CHECK(cudaError_t stmt)
  {
    int err_n = (int) (stmt);
    if (0 != err_n) {
      fprintf(stderr, "[%s:%d] CUDA call '%d' failed with %d: %s \n",
              __FILE__, __LINE__, stmt, err_n, cudaGetErrorString(stmt));
      exit(EXIT_FAILURE);
    }
    assert(cudaSuccess == err_n);

    return 0;
  }

#ifdef __cplusplus
}
#endif


#endif /* #define fillprint_h__ */
