#include <cudaDefs.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();


int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
}
