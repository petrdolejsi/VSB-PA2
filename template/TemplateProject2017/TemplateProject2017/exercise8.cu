#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <glew.h>
#include <freeglut.h>
#include <cudaDefs.h>
#include <imageManager.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

#include "imageKernels.cuh"
#include <random>

#define BLOCK_DIM 8

constexpr uint ThreadPerBlock = 512;
constexpr uint BlocksPerGrid = 1024;

constexpr size_t ArraySize = 1000000;
constexpr size_t ByteArraySize = ArraySize * sizeof(int);

__global__ void find_max(int *memory, int *max)
{
	__shared__ int sMax;
	sMax = 0;

	__syncthreads();

	uint tIdX = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint strideX = blockDim.x * gridDim.x;

	while (tIdX < ArraySize)
	{
		if (memory[tIdX] > sMax)
		{
			atomicMax(&sMax, memory[tIdX]);
		}

		tIdX += strideX;
	}

	__syncthreads();
	atomicMax(max, sMax);
}

void exercise8()
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> dist(-56464, 99994);

	int *host_randomArray;
	int *device_randomArray;
	host_randomArray = (int *)::operator new(ByteArraySize);

	for (size_t i = 0; i < ArraySize; i++)
	{
		host_randomArray[i] = dist(rd);
	}

	printf("Generated on cpu.\n");

	checkCudaErrors(cudaMalloc((void **)&device_randomArray, ByteArraySize));
	checkCudaErrors(cudaMemcpy(device_randomArray, host_randomArray, ByteArraySize, cudaMemcpyHostToDevice));
	free(host_randomArray);

	int *device_max;
	checkCudaErrors(cudaMalloc((void **)&device_max, sizeof(int)));

	find_max << <BlocksPerGrid, ThreadPerBlock >> > (device_randomArray, device_max);

	int max;
	checkCudaErrors(cudaMemcpy(&max, device_max, sizeof(int), cudaMemcpyDeviceToHost));

	printf("Found max value: %i\n", max);

	checkCudaErrors(cudaFree(device_max));
	checkCudaErrors(cudaFree(device_randomArray));
	
}