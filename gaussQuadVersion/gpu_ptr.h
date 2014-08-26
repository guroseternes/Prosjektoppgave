#ifndef GPU_PTR_H_
#define GPU_PTR_H_

//#include "Exception.hpp"
#include <cassert>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

/**
 * Very simple class that suits the GPU fine for accessing memory
 */
class gpu_raw_ptr {
public:
	float* ptr;   //!< pointer to allocated memory
	size_t pitch; //!< Pitch in bytes of allocated m
};

class gpu_ptr_2D {
public:
	// Allocating data on the GPU
	gpu_ptr_2D(unsigned int width, unsigned int height, int border = 0, float* cpu_ptr=NULL);

	//Deallocates the data
	~gpu_ptr_2D();

	const gpu_raw_ptr& getRawPtr() const {
		return data;
	}

	const unsigned int& getWidth() const {
		return data_width;
	}

	const unsigned int& getHeight() const {
                return data_height;
        }
        const int& getBorder() const {
                return data_border;
        }
	// Performs GPU-GPU copy of a width x height domain starting at x_offset, y_offset from a different gpu_ptr
	void copy(const gpu_ptr_2D& other,
			unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0, int border=0);

	void download(float* cpu_ptr,
			unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0);
	
	// Perform CPU to GPU copy of a witdh x height domain starting at x_offset and y_offset on the cpu_ptr.
	void upload(const float* cpu_ptr, unsigned int x_offset=0, unsigned int y_offset=0,
    		unsigned int width=0, unsigned int height=0);
	
	// Performs GPU "memset" of a width x height domain starting at x-offset, y_offset. Only really useful for setting all bits to 0 or 1.
	void set(int value, unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0, int border=0);

	// Same as above, only with float.
	void set(float value, unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0);

private:
	gpu_raw_ptr data;
	size_t pitch;
	unsigned int data_width;
	unsigned int  data_height;
	int data_border;
};

inline gpu_ptr_2D::gpu_ptr_2D(unsigned int width, unsigned int height, int border, float* cpu_ptr) {
	data_width = width + 2*border;
	data_height = height + 2*border;
	data_border = border;
	data.ptr = 0;
	data.pitch = 0;

	cudaMallocPitch((void**) &data.ptr, &data.pitch, data_width*sizeof(float), data_height);
	if (cpu_ptr != NULL) upload(cpu_ptr);
} 

inline gpu_ptr_2D::~gpu_ptr_2D() {
	cudaFree(data.ptr);
}

inline void gpu_ptr_2D::copy(const gpu_ptr_2D& other, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height, int border){
//	width = (width == 0) ? data_width : width;
//	height = (height == 0) ? data_height : height;

	data_width = width + 2*border;
	data_height = height + 2*border;
	data_border = border; 

	size_t pitch1 = data.pitch;
	float* ptr1 = (float*) ((char*) data.ptr+y_offset*pitch1) + x_offset;
 
	size_t pitch2 = other.getRawPtr().pitch; 
	float* ptr2 = (float*) ((char*) other.getRawPtr().ptr+y_offset*pitch2) + x_offset;

	assert(data_width == other.getWidth() && data_height == other.getHeight() && ptr1 != ptr2);

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyDeviceToDevice);
}

inline void gpu_ptr_2D::download(float* cpu_ptr, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height){
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	size_t pitch1 = width*sizeof(float);
	float* ptr1 = cpu_ptr;

	size_t pitch2 = data.pitch;
	float* ptr2 = (float*) ((char*) data.ptr+y_offset*pitch2) + x_offset;

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyDeviceToHost);
}

inline void gpu_ptr_2D::upload(const float* cpu_ptr, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height){	
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	size_t pitch1 = data.pitch;
	float* ptr1 = (float*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = width*sizeof(float);
	const float* ptr2 = cpu_ptr;

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyHostToDevice);
}

inline void gpu_ptr_2D::set(int value, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height, int border){	
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	data_width = width + 2*border;
	data_height = height + 2*border;

	data_border = border;

	size_t pitch = data.pitch;
	float* ptr = (float*) ((char*) data.ptr+y_offset*pitch) + x_offset;
	
	cudaMemset2D(ptr, pitch, value, width*sizeof(float), height);
}

inline void gpu_ptr_2D::set(float value, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height){	
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	std::vector<float> tmp(width*height, value);

	size_t pitch1 = data.pitch;
	float* ptr1 = (float*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = width*sizeof(float);
	const float* ptr2 = &tmp[0];

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyHostToDevice);
}

class gpu_ptr_1D{
public:
	gpu_ptr_1D(unsigned int width, float* cpu_ptr=NULL);
	~gpu_ptr_1D();
	//void download(float* cpu_ptr, unsigned int x_offset=0, unsigned int width=0);
	void upload(const float* cpu_ptr, unsigned int x_offset=0, unsigned int width=0);
	//void set(int value, unsigned int x_offset=0, unsigned int width=0);
	void set(float value, unsigned int x_offset=0, unsigned int width=0);
	float* getRawPtr() const {
		return data_ptr;
	}
	const unsigned int& getWidth() const {
		return data_width;
	}

private:
	float* data_ptr;
	unsigned int data_width;
};

inline void gpu_ptr_1D::upload(const float* cpu_ptr, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	float* ptr1 = data_ptr + x_offset;
	const float* ptr2 = cpu_ptr;

	cudaMemcpy(ptr1, ptr2, width*sizeof(float), cudaMemcpyHostToDevice);
}

inline gpu_ptr_1D::gpu_ptr_1D(unsigned int width, float* cpu_ptr) {
	data_width = width;
	data_ptr = 0;
	cudaMalloc((void**) &data_ptr, data_width*sizeof(float));
	if (cpu_ptr != NULL) upload(cpu_ptr);
}

inline gpu_ptr_1D::~gpu_ptr_1D() {
	cudaFree(data_ptr);
}

inline void gpu_ptr_1D::set(float value, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	float* ptr = data_ptr + x_offset;
	std::vector<float> tmp(width, value);

	cudaMemcpy(ptr, &tmp[0], width*sizeof(float), cudaMemcpyHostToDevice);
}

#endif
