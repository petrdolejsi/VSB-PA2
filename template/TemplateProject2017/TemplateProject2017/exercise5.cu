// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>


#include "imageKernels.cuh"

#define BLOCK_DIM 8

texture<float, 2, cudaReadModeElementType> tex_ref;
cudaChannelFormatDesc tex_channel_desc;

unsigned char *d_image_data = nullptr;
unsigned int image_width;
unsigned int image_height;
unsigned int image_bpp;		//Bits Per Pixel = 8, 16, 24, or 32 bit
unsigned int image_pitch;

size_t tex_pitch;
float *d_linear_pitch_texture_data = nullptr;
cudaArray *d_array_texture_data = nullptr;

KernelSetting ks;

float *d_output_data = nullptr;

void load_source_image(const char* image_file_name)
{
	FreeImage_Initialise();
	const auto tmp = ImageManager::GenericLoader(image_file_name, 0);

	image_width = FreeImage_GetWidth(tmp);
	image_height = FreeImage_GetHeight(tmp);
	image_bpp = FreeImage_GetBPP(tmp);
	image_pitch = FreeImage_GetPitch(tmp);

	cudaMalloc(reinterpret_cast<void**>(&d_image_data), image_pitch * image_height * image_bpp / 8);
	cudaMemcpy(d_image_data, FreeImage_GetBits(tmp), image_pitch * image_height * image_bpp / 8, cudaMemcpyHostToDevice);

	checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), image_pitch, image_height, image_width, "%hhu ", "Result of Linear Pitch Text");
	checkDeviceMatrix<unsigned char>(d_image_data, image_pitch, image_height, image_width, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

void create_texture_from_linear_pitch_memory()
{
	//Floating Point Texture Data
	cudaMallocPitch(reinterpret_cast<void**>(&d_linear_pitch_texture_data), &tex_pitch, image_width * sizeof(float), image_height);

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch (image_bpp)
	{
	case 8:  colorToFloat<8 > << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 16: colorToFloat<16> << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 24: colorToFloat<24> << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 32: colorToFloat<32> << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	}

	checkDeviceMatrix<float>(d_linear_pitch_texture_data, tex_pitch, image_height, image_width, "%6.1f ", "Result of Linear Pitch Text");

	//Texture settings
	tex_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	tex_ref.normalized = false;
	tex_ref.filterMode = cudaFilterModePoint;
	tex_ref.addressMode[0] = cudaAddressModeClamp;
	tex_ref.addressMode[1] = cudaAddressModeClamp;

	cudaBindTexture2D(nullptr, &tex_ref, d_linear_pitch_texture_data, &tex_channel_desc, image_width, image_height, tex_pitch);
}

void createTextureFrom2DArray()
{
	//Texture settings
	tex_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	tex_ref.normalized = false;
	tex_ref.filterMode = cudaFilterModePoint;
	tex_ref.addressMode[0] = cudaAddressModeClamp;
	tex_ref.addressMode[1] = cudaAddressModeClamp;

	//Converts custom image data to float and stores result in the float_linear_data
	float *d_linear_texture_data = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&d_linear_texture_data), image_width * image_height * sizeof(float));
	switch (image_bpp)
	{
	case 8:  colorToFloat<8 > << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_width, d_linear_texture_data); break;
	case 16: colorToFloat<16> << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_width, d_linear_texture_data); break;
	case 24: colorToFloat<24> << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_width, d_linear_texture_data); break;
	case 32: colorToFloat<32> << <ks.dimGrid, ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_width, d_linear_texture_data); break;
	}
	cudaMallocArray(&d_array_texture_data, &tex_channel_desc, image_width, image_height);
	cudaMemcpyToArray(d_array_texture_data, 0, 0, d_linear_texture_data, image_width * image_height * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(&tex_ref, d_array_texture_data, &tex_channel_desc);

	cudaFree(d_linear_texture_data);
}


void release_memory()
{
	cudaUnbindTexture(tex_ref);
	if (d_image_data != nullptr)
		cudaFree(d_image_data);
	if (d_linear_pitch_texture_data != nullptr)
		cudaFree(d_linear_pitch_texture_data);
	if (d_array_texture_data)
		cudaFreeArray(d_array_texture_data);
	if (d_output_data)
		cudaFree(d_output_data);
}


__global__ void tex_kernel(const unsigned int tex_width, const unsigned int tex_height, float* dst)
{
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	if ((tx < tex_width) && (ty < tex_height))
	{
		dst[ty * tex_width + tx] = tex2D(tex_ref, tx, ty);
	}
}


void exercise5()
{
	//initializeCUDA(deviceProp);

	load_source_image("terrain10x10.tif");

	cudaMalloc(reinterpret_cast<void**>(&d_output_data), image_width * image_height * sizeof(float));

	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimGrid = dim3((image_width + BLOCK_DIM - 1) / BLOCK_DIM, (image_height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	//Test 1 - texture stored in linear pitch memory
	create_texture_from_linear_pitch_memory();
	tex_kernel << <ks.dimGrid, ks.dimBlock >> > (image_width, image_height, d_output_data);
	checkDeviceMatrix<float>(d_output_data, image_width * sizeof(float), image_height, image_width, "%6.1f ", "dOutputData");

	//Test 2 - texture stored in 2D array
	createTextureFrom2DArray();
	tex_kernel << <ks.dimGrid, ks.dimBlock >> > (image_width, image_height, d_output_data);
	checkDeviceMatrix<float>(d_output_data, image_width * sizeof(float), image_height, image_width, "%6.1f ", "dOutputData");

	release_memory();
}