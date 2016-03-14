#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static void checkCudaCall(cudaError_t error, const char* file, int line)
{
	if (error)
	{
		std::cout << "CUDA error at " << file << ":" << line << std::endl;
		std::cout << cudaGetErrorName(error) << " :: " << cudaGetErrorString(error) << std::endl;
		__debugbreak();
	}
}
#define CHECK_CUDA_CALL(err) (checkCudaCall(err, __FILE__, __LINE__))

/*
 * Represents RAII CUDA memory allocated on (GPU) device.
 */
template <typename T>
class CudaMemory
{
public:
	CudaMemory(size_t count = 1, T* mem = nullptr) : count(count)
	{
		CHECK_CUDA_CALL(cudaMalloc(&this->devicePtr, sizeof(T) * count));

		if (mem)
		{
			this->store(*mem, count);
		}
	}
	CudaMemory(size_t count, T value) : count(count)
	{
		CHECK_CUDA_CALL(cudaMalloc(&this->devicePtr, sizeof(T) * count));
		CHECK_CUDA_CALL(cudaMemset(this->devicePtr, value, sizeof(T) * count));
	}
	~CudaMemory()
	{
		CHECK_CUDA_CALL(cudaFree(this->devicePtr));
		this->devicePtr = nullptr;
	}

	CudaMemory(const CudaMemory& other) = delete;
	CudaMemory& operator=(const CudaMemory& other) = delete;
	CudaMemory(CudaMemory&& other) = delete;

	T* operator*()
	{
		return this->devicePtr;
	}

	void load(T& dest, size_t count = 1) const
	{
		if (count == 0)
		{
			count = this->count;
		}

		CHECK_CUDA_CALL(cudaMemcpy(&dest, this->devicePtr, sizeof(T) * count, cudaMemcpyDeviceToHost));
	}
	void store(const T& src, size_t count = 1, size_t start_index = 0)
	{
		if (count == 0)
		{
			count = this->count;
		}

		CHECK_CUDA_CALL(cudaMemcpy(this->devicePtr + start_index, &src, sizeof(T) * count, cudaMemcpyHostToDevice));
	}

private:
	T* devicePtr = nullptr;
	size_t count;
};

/*
 * Represents RAII CUDA pinned memory allocated on (CPU) host, directly mapped and accessible from device.
 * Requires 64-bit CUDA context and cudaDeviceMapHost flag to be set.
 */
template <typename T>
class CudaHostMemory
{
public:
	CudaHostMemory(size_t count = 1) : count(count)
	{
		cudaMallocHost(&hostPointer, sizeof(T) * count);
		cudaHostGetDevicePointer(&this->devicePointer, this->hostPointer, 0);
	}
	~CudaHostMemory()
	{
		CHECK_CUDA_CALL(cudaFreeHost(this->hostPointer));
		this->hostPointer = nullptr;
		this->devicePointer = nullptr;
	}

	T* host() const
	{
		return this->hostPointer;
	}
	T* device() const
	{
		return this->devicePointer;
	}

private:
	T* hostPointer;
	T* devicePointer;
	size_t count;
};
