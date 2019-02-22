#include <cudaDefs.h>
#include "exercise1.h"


cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();


int main(int argc, char *argv[])
{
	//initializeCUDA(deviceProp);
	exercise1();

	return 0;
}
