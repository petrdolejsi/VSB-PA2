#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>
#include "exercise3.cuh"

constexpr unsigned int TPB = 128;
constexpr unsigned int NO_FORCES = 256;
constexpr unsigned int NO_RAIN_DROPS = 1 << 20;

constexpr unsigned int MEM_BLOCKS_PER_THREAD_BLOCK = 8;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

void exercise3()
{
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	float3 *hForces = createData(NO_FORCES);
	float3 *hDrops = createData(NO_RAIN_DROPS);

	float3 *dForces = nullptr;
	float3 *dDrops = nullptr;
	float3 *dFinalForce = nullptr;

	error = cudaMalloc((void**)&dForces, NO_FORCES * sizeof(float3));
	error = cudaMemcpy(dForces, hForces, NO_FORCES * sizeof(float3), cudaMemcpyHostToDevice);

	error = cudaMalloc((void**)&dDrops, NO_RAIN_DROPS * sizeof(float3));
	error = cudaMemcpy(dDrops, hDrops, NO_RAIN_DROPS * sizeof(float3), cudaMemcpyHostToDevice);

	error = cudaMalloc((void**)&dFinalForce, sizeof(float3));

	KernelSetting ksReduce;

	//TODO: ... Set ksReduce
	ksReduce.dimBlock = dim3(TPB, 1, 1);
	ksReduce.dimGrid = dim3(1, 1, 1);


	KernelSetting ksAdd;
	//TODO: ... Set ksAdd
	ksAdd.dimBlock = dim3(TPB, 1, 1);
	ksAdd.dimGrid = dim3(1, 1, 1);

	for (unsigned int i = 0; i < 1000; i++)
	{
		reduce << <ksReduce.dimGrid, ksReduce.dimBlock >> > (dForces, NO_FORCES, dFinalForce);
		add << <ksAdd.dimGrid, ksAdd.dimBlock >> > (dFinalForce, NO_RAIN_DROPS, dDrops);
	}

	checkDeviceMatrix<float>((float*)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
	// checkDeviceMatrix<float>((float*)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

	if (hForces)
		free(hForces);
	if (hDrops)
		free(hDrops);

	cudaFree(dForces);
	cudaFree(dDrops);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	printf("Time to get device properties: %f ms", elapsedTime);
}

__global__ void reduce(const float3 * __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce)
{
	__shared__ float3 sForces[TPB];					//SEE THE WARNING MESSAGE !!!
	unsigned int tid = threadIdx.x;
	unsigned int next = TPB;						//SEE THE WARNING MESSAGE !!!

	//TODO: Make the reduction

	float3 *src = &sForces[tid];
	float3 *src2 = (float3 *)&dForces[tid + next];

	*src = dForces[tid];

	src->x += src2->x;
	src->y += src2->y;
	src->z += src2->z;

	__syncthreads();

	next >>= 2;

	if (tid >= next)
	{
		return;
	}
	src2 = src + next;
}

__global__ void add(const float3* __restrict__ dFinalForce, const unsigned int noRainDrops, float3* __restrict__ dRainDrops)
{
	//TODO: Add the FinalForce to every Rain drops position.

	unsigned int bid = blockIdx.x * MEM_BLOCKS_PER_THREAD_BLOCK + threadIdx.x;
}

float3 *createData(const unsigned int length)
{
	random_device rd;
	mt19937_64 mt(rd());
	uniform_int_distribution<float> dist(0.0f, 1.0f);
	
	//TODO: Generate float3 vectors. You can use 'make_float3' method.
	float3 *data = static_cast<float3*>(::operator new(sizeof(float3) * length));

	for (unsigned int i = 0; i < length; i++)
	{
		//data[i] = make_float3(dist(mt), dist(mt), dist(mt));
		data[i] = make_float3(1.0f, 1.0f, 1.0f);
	}
		
	return data;
}

void printData(const float3 *data, const unsigned int length)
{
	if (data == 0) return;
	const float3 *ptr = data;
	for (unsigned int i = 0; i < length; i++, ptr++)
	{
		printf("%5.2f %5.2f %5.2f ", ptr->x, ptr->y, ptr->z);
	}
}