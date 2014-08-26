#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_
#include <cuda_runtime.h>
#include "gpu_ptr.h"


/**
  * Calculates the address based on input parameters
  */

__host__ float* address2D(float* base, unsigned int pitch, unsigned int x, unsigned int y) {
        return (float*) ((char*) base+y*pitch) + x;
}

/**
  * Calculates the address based on input parameters
  * @param base The start pointer of the array
  * @param pitch number of bytes in each row
  * @param x offset in elements to x direction
  * @param y offset in elements in y direction
  */

__device__ float* device_address2D(float* base, unsigned int pitch, unsigned int x, unsigned int y) {
        return (float*) ((char*) base+y*pitch) + x;
}

__device__ float* global_index(float* base, unsigned int pitch, int x, int y, int border = 0) {
        return (float*) ((char*) base+(y+border)*pitch) + (x+border);
}
#endif

