#include <cudaDefs.h>
#include "exercise2.cuh"

const size_t Rows = 150;
const size_t Cols = 200;
const size_t BlockSize = 8;

void exercise2()
{
	int *dMatrix;
	size_t pitchInBytes = 0;
	checkCudaErrors(cudaMallocPitch((void**)&dMatrix, &pitchInBytes, Cols * sizeof(int), Rows));

	size_t pitch = pitchInBytes / sizeof(int);
	dim3 grid = dim3(getNumberOfParts(Rows, BlockSize), getNumberOfParts(Cols, BlockSize));
	dim3 block = dim3(BlockSize, BlockSize);

	fill << <grid, block >> > (dMatrix, Rows, Cols, pitch);
	checkDeviceMatrix(dMatrix, pitchInBytes, Rows, Cols, "%-3d ", "dMatrix");

	increment << <grid, block >> > (dMatrix, Rows, Cols, pitch);
	checkDeviceMatrix(dMatrix, pitchInBytes, Rows, Cols, "%-3d ", "dMatrix");

	int *expectedMatrix = new int[Rows * Cols];
	for (size_t i = 0; i < Rows * Cols; i++)
		expectedMatrix[i] = i + 1;

	int *matrix = new int[pitch * Rows];
	checkCudaErrors(cudaMemcpy2D(matrix, pitchInBytes, dMatrix, pitchInBytes, Cols * sizeof(int), Rows, cudaMemcpyDeviceToHost));
	checkHostMatrix(matrix, pitchInBytes, Rows, Cols, "%-3d ", "matrix");

	delete[] matrix;
	delete[] expectedMatrix;
	cudaFree(dMatrix);
}

__global__ void fill(int* matrix, size_t rows, size_t cols, size_t pitch)
{
	int row = blockIdx.x * BlockSize + threadIdx.x;
	int col = blockIdx.y * BlockSize + threadIdx.y;
	if (row >= rows || col >= cols)
	{
		return;
	}

	matrix[row * pitch + col] = col * rows + row;
}

__global__ void increment(int* matrix, size_t rows, size_t cols, size_t pitch)
{
	int row = blockIdx.x * BlockSize + threadIdx.x;
	int col = blockIdx.y * BlockSize + threadIdx.y;
	if (row >= rows || col >= cols)
	{
		return;
	}
		

	matrix[row * pitch + col]++;
}