#include <cudaDefs.h>
#include <time.h>
#include <math.h>


const unsigned int N = 1 << 20;
const unsigned int MEMSIZE = N * sizeof(unsigned int);
const unsigned int NO_LOOPS = 100;
const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int GRID_SIZE = (N + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;

void fillData(unsigned int *data, const unsigned int length)
{
	//srand(time(0));
	for (unsigned int i=0; i<length; i++)
	{
		//data[i]= rand();
		data[i]= 1;
	}
}

void printData(const unsigned int *data, const unsigned int length)
{
	if (data ==0) return;
	for (unsigned int i=0; i<length; i++)
	{
		printf("%u ", data[i]);
	}
}


__global__ void kernel(const unsigned int *a, const unsigned int *b, const unsigned int length, unsigned int *c)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//TODO:  thread block loop
	if (tid < length)
	{
		c[tid] = a[tid] + b[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test1()
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	//T ODO: create stream
	
	unsigned int dataOffset = 0;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	for(int i=0; i < NO_LOOPS; i++)
	{
		//TODO:  copy a->da, b->db
		//TODO:  run the kernel in the stream
		//TODO:  copy dc->c
		cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream >> > (da, db, N, dc);
		cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream);

		dataOffset += N;
	}

	//TODO: Synchonize stream

	cudaStreamSynchronize(stream);

	//TODO: Destroy stream

	cudaStreamDestroy(stream);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test 1 time: %f ms\n", elapsedTime);

	printData(c, 100);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test2()
{
	//TODO: reuse the source code of above mentioned method test1()
	cudaStream_t stream0;
	cudaStream_t stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	//T ODO: create stream

	unsigned int dataOffset0 = 0;
	unsigned int dataOffset1 = N;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	for (int i = 0; i < NO_LOOPS; i+=2)
	{
		//TODO:  copy a->da, b->db
		//TODO:  run the kernel in the stream
		//TODO:  copy dc->c
		cudaMemcpyAsync(da, &a[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(db, &b[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream0 >> > (da, db, N, dc);
		cudaMemcpyAsync(&c[dataOffset0], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream0);

		dataOffset0 += (2 * N);

		cudaMemcpyAsync(da, &a[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(db, &b[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream1 >> > (da, db, N, dc);
		cudaMemcpyAsync(&c[dataOffset1], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream1);

		dataOffset1 += (2 * N);
	}

	//TODO: Synchonize stream

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	//TODO: Destroy stream

	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("\nTest 2 time: %f ms\n", elapsedTime);

	printData(c, 100);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test3()
{
	//TODO: reuse the source code of above mentioned method test1()
	cudaStream_t stream0;
	cudaStream_t stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	//T ODO: create stream

	unsigned int dataOffset0 = 0;
	unsigned int dataOffset1 = N;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	for (int i = 0; i < NO_LOOPS; i += 2)
	{
		//TODO:  copy a->da, b->db
		//TODO:  run the kernel in the stream
		//TODO:  copy dc->c
		cudaMemcpyAsync(da, &a[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(da, &a[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);

		cudaMemcpyAsync(db, &b[dataOffset0], MEMSIZE, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(db, &b[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1);

		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream0 >> > (da, db, N, dc);
		kernel << <GRID_SIZE, THREAD_PER_BLOCK, 0, stream1 >> > (da, db, N, dc);
		
		cudaMemcpyAsync(&c[dataOffset0], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(&c[dataOffset1], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream1);

		dataOffset0 += (2 * N);
		dataOffset1 += (2 * N);
	}

	//TODO: Synchonize stream

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	//TODO: Destroy stream

	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("\nTest 3 time: %f ms\n", elapsedTime);

	printData(c, 100);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}


void exercise10()
{

	test1();
	test2();
	test3();

}
