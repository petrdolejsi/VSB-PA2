#include <cudaDefs.h>
#include "exercise1.cuh"
#include "exercise2.cuh"
#include "exercise3.cuh"

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

void exercise5();
void exercise6();
void exercise7(int argc, char *argv[]);
void project_color();
void project_gray();
void exercise8();
void exercise10();
void exercise11();
void project(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	//exercise1();
	//exercise2();
	//exercise3();
	//exercise3();
	//exercise5();
	//exercise6();
	//exercise7(argc, argv);
	//exercise8();
	//project_color();
	project(argc, argv);
	//project_gray();
	//exercise10();
	//exercise11();

	std::getchar();

	return 0;
}
