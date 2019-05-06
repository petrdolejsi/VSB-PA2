#include <cudaDefs.h>

typedef struct Cube {
	int x;
	int y;
	int z;
} Cube;

__constant__ __device__ int d_scalar;
__constant__ __device__ struct Cube d_cube;
__constant__ __device__ int d_array[5];

void exercise4()
{
	// int
	
	int h_scalar = 89;
	cudaMemcpyToSymbol(static_cast<const void*>(&d_scalar), static_cast<const void*>(&h_scalar), sizeof(int));

	int h_scalar_returned;
	cudaMemcpyFromSymbol(static_cast<void*>(&h_scalar_returned), static_cast<const void*>(&d_scalar), sizeof(int));

	checkError();

	printf("Scalar value (int): %d\n", h_scalar_returned);

	// cube

	Cube h_cube;
	h_cube.x = 10;
	h_cube.y = 20;
	h_cube.z = 30;
	cudaMemcpyToSymbol(static_cast<const void*>(&d_cube), static_cast<const void*>(&h_cube), sizeof(Cube));

	Cube h_cube_returned;
	cudaMemcpyFromSymbol(static_cast<void*>(&h_cube_returned), static_cast<const void*>(&d_cube), sizeof(Cube));

	checkError();

	printf("Cube - x: %d, y: %d, z: %d\n", h_cube_returned.x, h_cube_returned.y, h_cube_returned.z);

	// array

	int h_array[5] = {1,2,3,4,5};
	cudaMemcpyToSymbol(static_cast<const void*>(&d_array), static_cast<const void*>(&h_array), sizeof(int) * 5);

	int h_array_returned[5];
	cudaMemcpyFromSymbol(static_cast<void*>(&h_array_returned), static_cast<const void*>(&d_array), sizeof(int) * 5);

	checkError();

	printf("Array - [0]: %d, [1]: %d, [2]: %d, [3]: %d, [4]: %d\n", h_array_returned[0], h_array_returned[1], h_array_returned[2], h_array_returned[3], h_array_returned[4]);
}