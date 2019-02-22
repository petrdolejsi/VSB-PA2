#include <cudaDefs.h>
#include "exercise1.h"

void exercise1()
{
	part1_1();
	part1_2();
}

void part1_1()
{
	printf("Exercise 1 - Part 1\n");
	
	const unsigned int m = 10;
	const unsigned int size = m * sizeof(int);

	auto a_host = (int*)malloc(size);
	auto b_host = (int*)malloc(size);
	auto c_host = (int*)malloc(size);

	for (auto i = 0; i < m; i++)
	{
		a_host[i] = 2 * i;
		b_host[i] = 3 * i;
	}

	int *a_device;
	int *b_device;
	int *c_device;

	auto err = cudaMalloc(&a_device, size);
	if (err != cudaSuccess)
	{
		printf("Error in allocating device A\n");
		exit(1);
	}
	err = cudaMalloc(&b_device, size);
	if (err != cudaSuccess)
	{
		printf("Error in allocating device B\n");
		exit(1);
	}
	err = cudaMalloc(&c_device, size);
	if (err != cudaSuccess)
	{
		printf("Error in allocating device C\n");
		exit(1);
	}

	err = cudaMemcpy(a_device, a_host, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("Error in copying data to device A\n");
		exit(1);
	}
	err = cudaMemcpy(b_device, b_host, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("Error in copying data to device B\n");
		exit(1);
	}

	vector_add_m <<< 1, m >>> (a_device, b_device, c_device, m);

	err = cudaMemcpy(c_host, c_device, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		printf("Error in getting data from device C\n");
		exit(1);
	}

	for (auto i = 0; i < m; i++)
	{
		std::cout << c_host[i] << std::endl;
	}

	err = cudaFree(a_device);
	if (err != cudaSuccess)
	{
		printf("Error in freeing device A\n");
		exit(1);
	}
	err = cudaFree(b_device);
	if (err != cudaSuccess)
	{
		printf("Error in freeing device B\n");
		exit(1);
	}
	err = cudaFree(c_device);
	if (err != cudaSuccess)
	{
		printf("Error in freeing deviceC\n");
		exit(1);
	}

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
	
	const unsigned int m = 10;
	const unsigned int n = 10;

	auto *a_host = (int*)malloc(m * n * sizeof(int));
	auto *b_host = (int*)malloc(m * n * sizeof(int));
	auto *c_host = (int*)malloc(m * n * sizeof(int));

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

	auto err = cudaMalloc(&a_device, sizeof(int) * m * n);
	if (err != cudaSuccess)
	{
		printf("Error in allocating row\n");
		exit(1);
	}

	err = cudaMalloc(&b_device, sizeof(int) * m * n);
	if (err != cudaSuccess)
	{
		printf("Error in allocating row\n");
		exit(1);
	}

	err = cudaMalloc(&c_device, sizeof(int) * m * n);
	if (err != cudaSuccess)
	{
		printf("Error in allocating row\n");
		exit(1);
	}
	
	err = cudaMemcpy(a_device, a_host, m * n * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("Error in copying data to device A\n");
		exit(1);
	}
	err = cudaMemcpy(b_device, b_host, m * n * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("Error in copying data to device B\n");
		exit(1);
	}

	dim3 dimBlock(m, n);
	dim3 dimGrid(1, 1);

	vector_add_n_m <<<dimGrid, dimBlock >>> (a_device, b_device, c_device, m, n);

	err = cudaMemcpy(c_host, c_device, m * n * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		printf("Error in getting data from device C\n");
		exit(1);
	}

	for (int i = 0; i < m; i++)
	{
		std::cout << i << ": " ;
		for (int j = 0; j < n; j++)
		{
			std::cout << c_host[i * m + j] << "  ";
		}
		std::cout << std::endl;
	}

	err = cudaFree(a_device);
	if (err != cudaSuccess)
	{
		printf("Error in freeing deviceC\n");
		exit(1);
	}
	err = cudaFree(b_device);
	if (err != cudaSuccess)
	{
		printf("Error in freeing deviceC\n");
		exit(1);
	}
	err = cudaFree(c_device);
	if (err != cudaSuccess)
	{
		printf("Error in freeing deviceC\n");
		exit(1);
	}

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
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < m && j < n)
	{
		c[i * m + j] = a[i * m + j] + b[i * m + j];
	}
}