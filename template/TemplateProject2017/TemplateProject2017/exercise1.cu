#include <cudaDefs.h>
#include <time.h>
#include "exercise1.cuh"

constexpr unsigned int THREADS_PER_BLOCK = 256;
constexpr unsigned int MEMBLOCK_PER_THREADBLOCK = 2;

void exercise1()
{
	part1_1();
	//part1_2();
}

void part1_1()
{
	srand(time(NULL));
	
	printf("Exercise 1 - Part 1\n");
	
	const unsigned int m = 5000;
	const unsigned int size = m * sizeof(int);

	auto a_host = static_cast<int*>(malloc(size));
	auto b_host = static_cast<int*>(malloc(size));
	auto c_host = static_cast<int*>(malloc(size));

	for (auto i = 0; i < m; i++)
	{
		a_host[i] = rand();
		b_host[i] = rand();
	}

	int *a_device;
	int *b_device;
	int *c_device;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	checkCudaErrors(cudaMalloc(&a_device, size));
	checkCudaErrors(cudaMalloc(&b_device, size));
	checkCudaErrors(cudaMalloc(&c_device, size));

	checkCudaErrors(cudaMemcpy(a_device, a_host, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_device, b_host, size, cudaMemcpyHostToDevice));

	vector_add_m <<< 1, m >>> (a_device, b_device, c_device, m);

	checkCudaErrors(cudaMemcpy(c_host, c_device, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(a_device));
	checkCudaErrors(cudaFree(b_device));
	checkCudaErrors(cudaFree(c_device));

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	for (auto i = 0; i < m; i++)
	{
		std::cout << c_host[i] << std::endl;
	}

	printf("Time to get device properties: %f ms", elapsedTime);

	free(a_host);
	free(b_host);
	free(c_host);

	a_host = nullptr;
	b_host = nullptr;
	c_host = nullptr;
}

void part1_2()
{
	printf("Exercise 1 - Part 2\n");
	
	const unsigned int m = 20;
	const unsigned int n = 20;

	auto *a_host = static_cast<int*>(malloc(m * n * sizeof(int)));
	auto *b_host = static_cast<int*>(malloc(m * n * sizeof(int)));
	auto *c_host = static_cast<int*>(malloc(m * n * sizeof(int)));

	for (auto i = 0; i < m; i++)
	{
		for (auto j = 0; j < n; j++)
		{
			a_host[i * m + j] = (i + 1) * (j + 1);
			b_host[i * m + j] = (i + 1) * (j + 1);
		}
	}

	int *a_device;
	int *b_device;
	int *c_device;

	checkCudaErrors(cudaMalloc(&a_device, sizeof(int) * m * n));
	checkCudaErrors(cudaMalloc(&b_device, sizeof(int) * m * n));
	checkCudaErrors(cudaMalloc(&c_device, sizeof(int) * m * n));
	
	checkCudaErrors(cudaMemcpy(a_device, a_host, m * n * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_device, b_host, m * n * sizeof(int), cudaMemcpyHostToDevice));

	dim3 dimBlock(m, n);
	dim3 dimGrid(1, 1);

	vector_add_n_m <<<dimGrid, dimBlock >>> (a_device, b_device, c_device, m, n);

	checkCudaErrors(cudaMemcpy(c_host, c_device, m * n * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < m; i++)
	{
		std::cout << i << ": " ;
		for (int j = 0; j < n; j++)
		{
			std::cout << c_host[i * m + j] << "  ";
		}
		std::cout << std::endl;
	}

	checkCudaErrors(cudaFree(a_device));
	checkCudaErrors(cudaFree(b_device));
	checkCudaErrors(cudaFree(c_device));

	free(a_host);
	free(b_host);
	free(c_host);

	a_host = nullptr;
	b_host = nullptr;
	c_host = nullptr;
}

__global__ void vector_add_m (int *a, int *b, int *c, const int m)
{
	const int i = threadIdx.x;
	if (i < m)
	{
		c[i] = a[i] + b[i];
	}
}

__global__ void vector_add_n_m (int *a, int *b, int *c, const int m, const int n)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	const auto j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < m && j < n)
	{
		c[i * m + j] = a[i * m + j] + b[i * m + j];
	}
}