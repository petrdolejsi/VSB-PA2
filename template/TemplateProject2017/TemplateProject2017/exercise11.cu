#include <cudaDefs.h>
#include <cublas_v2.h>

//cudaError_t error = cudaSuccess;
//cudaDeviceProp deviceProp = cudaDeviceProp();

cublasStatus_t status = cublasStatus_t();
cublasHandle_t handle = cublasHandle_t();

const unsigned int N = 5;
const unsigned int dim = 3;
const unsigned int MEMSIZE = N * dim * sizeof(float);
const unsigned int THREAD_PER_BLOCK = 128;
const unsigned int GRID_SIZE = (N * dim + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;

void fillData(float *data, const unsigned int length, const unsigned int dim)
{
	unsigned int id = 0;
	for (unsigned int i=0; i<length; i++)
	{
		for (unsigned int j=0; j<dim; j++)
		{
			data[id++]= i & 255;   //=i%256
		}
	}
}

void fillDataWithNumber(float *data, const unsigned int length, const unsigned int dim, const float number)
{
	unsigned int id = 0;
	for (unsigned int i=0; i<length; i++)
	{
		for (unsigned int j=0; j<dim; j++)
		{
			data[id++]= number;
		}
	}
}

__global__ void kernelPowerTwo(const float *a, const float *b, const unsigned int length, float *a2, float *b2)
{
//TODO:
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);
	
	a2[row * length + col] = a[row * length + col] * a[row * length + col];
	b2[row * length + col] = b[row * length + col] * b[row * length + col];
}

void exercise11()
{
	//initializeCUDA(deviceProp);
	status = cublasCreate(&handle) ;
		
	float alpha, beta;
	float *a, *b, *m;
	float *da, *da2, *db, *db2, *dm;
	float *ones, *dones;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&ones, MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&m, N * N * sizeof(float),cudaHostAllocDefault);

	cudaMalloc( (void**)&da, MEMSIZE );
	cudaMalloc( (void**)&da2, MEMSIZE );
	cudaMalloc( (void**)&db, MEMSIZE );
	cudaMalloc( (void**)&db2, MEMSIZE );
	cudaMalloc( (void**)&dones, MEMSIZE );
	cudaMalloc( (void**)&dm, N * N * sizeof(float));

	fillData(a, N, dim);
	fillData(b, N, dim);
	fillDataWithNumber(ones, N, dim, 1.0f);
	
	//Copy data to DEVICE
	cudaMemcpy(da, a, MEMSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, MEMSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dones, ones, MEMSIZE, cudaMemcpyHostToDevice);

	//TODO 1: Process a -> a^2  and b->b^2
	kernelPowerTwo<<<GRID_SIZE, THREAD_PER_BLOCK>>>(da, db, N * dim, da2, db2);

	//TODO 2: Process a^2 + b^2 using CUBLAS //pair-wise operation such that the result is dm[N*N] matrix

	float zero = 0.0f;
	float one = 1.0f;
	float min_two = -2.0f;
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, dim, &one, da2, dim, dones, dim, &zero, dm, N);
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, dim, &one, dones, dim, db2, dim, &one, dm, N);

	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, dim, &min_two, da, dim, db, dim, &one, dm, N);
		
	//TODO 3: Process -2ab and sum with previous result stored in dm using CUBLAS


	checkDeviceMatrix<float>(da,	sizeof(float)*dim, N, dim, "%f ", "A");
	checkDeviceMatrix<float>(da2,	sizeof(float)*dim, N, dim, "%f ", "A^2");
	checkDeviceMatrix<float>(db,	sizeof(float)*dim, N, dim, "%f ", "B");
	checkDeviceMatrix<float>(db2,   sizeof(float)*dim, N, dim, "%f ", "B^2");
	checkDeviceMatrix<float>(dones, sizeof(float)*dim, N, dim, "%f ", "ONES");
	checkDeviceMatrix<float>(dm,	sizeof(float)*N, N, N, "%f ", "M");

	
	cudaFree(da);
	cudaFree(da2);
	cudaFree(db);
	cudaFree(db2);
	cudaFree(dm);
	cudaFree(dones);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(m);
	cudaFreeHost(ones);

	status = cublasDestroy(handle);
}
