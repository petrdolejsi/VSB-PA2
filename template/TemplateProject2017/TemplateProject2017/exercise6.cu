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

uchar3 *dst_tex_data;

KernelSetting square_ks;

float *d_output_data = nullptr;

__constant__  int sobel_x_filter[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
__constant__  int sobel_y_filter[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

template<bool NormalizeTexel>__global__ void float_heighmap_texture_to_normalmap(const unsigned int tex_width, const unsigned int tex_height, const unsigned int dst_pitch, uchar3* dst)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	float x = 0, y = 0, z = 0;

	z = 0.5;
	const auto offset = col + row * (dst_pitch / 3);

	for (unsigned int i = 0; i < 3; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			const float texel = tex2D(tex_ref, col + (j - 1), row + (i - 1));
			x += texel * sobel_x_filter[j + i * 3];
			y += texel * sobel_y_filter[j + i * 3];
		}
	}
	x = x / 9;
	y = y / 9;

	if (NormalizeTexel) {
		const auto distance = sqrt(x * x + y * y + z * z);
		x /= distance;
		y /= distance;
		z /= distance;
	}

	uchar3 rgb_texel;
	uchar3 bgr_texel;
	rgb_texel.x = (x + 1) * 127.5;
	rgb_texel.y = (y + 1) * 127.5;
	rgb_texel.z = z * 255;

	bgr_texel.x = rgb_texel.z;
	bgr_texel.y = rgb_texel.y;
	bgr_texel.z = rgb_texel.x;

	dst[offset] = rgb_texel;
}

#pragma region STEP 1

//TASK:	Load the input image and store loaded data in DEVICE memory (dSrcImageData)

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

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
#pragma endregion

#pragma region STEP 2

//TASK: Create a texture based on the source image. The input images can have variable BPP (Byte Per Pixel), but finally any such image will be converted into the floating-point texture using
//		the colorToFloat kernel.

void create_src_texure()
{
	//Floating Point Texture Data
	cudaMallocPitch(reinterpret_cast<void**>(&d_linear_pitch_texture_data), &tex_pitch, image_width * sizeof(float), image_height);

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch (image_bpp)
	{
	case 8:  colorToFloat<8> << <square_ks.dimGrid, square_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 16: colorToFloat<16> << <square_ks.dimGrid, square_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 24: colorToFloat<24> << <square_ks.dimGrid, square_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 32: colorToFloat<32> << <square_ks.dimGrid, square_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	}

	//checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "", "");

	//Texture settings
	tex_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	tex_ref.normalized = false;
	tex_ref.filterMode = cudaFilterModePoint;
	tex_ref.addressMode[0] = cudaAddressModeClamp;
	tex_ref.addressMode[1] = cudaAddressModeClamp;

	cudaBindTexture2D(nullptr, &tex_ref, d_linear_pitch_texture_data, &tex_channel_desc, image_width, image_height, tex_pitch);
}
#pragma endregion

#pragma region STEP 3

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

void create_normal_map()
{
	//T ODO: Allocate Pitch memory dstTexData to store output texture
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&dst_tex_data), &tex_pitch, image_width * sizeof(float), image_height));

	//T ODO: Call the kernel that creates the normal map.
	float_heighmap_texture_to_normalmap<true> << <square_ks.dimGrid, square_ks.dimBlock >> > (image_width, image_height, tex_pitch, dst_tex_data);

	//check_data<uchar3>::checkDeviceMatrix(dstTexData, imageHeight, texPitch / sizeof(uchar3), true, "%hhu %hhu %hhu | ", "Result of Linear Pitch Text");
}

#pragma endregion

#pragma region STEP 4

//TASK: Save output image (normal map)

void save_tex_image(const char* image_file_name)
{
	FreeImage_Initialise();

	const auto tmp = FreeImage_Allocate(image_width, image_height, 24);
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dst_tex_data, tex_pitch, image_width * 3, image_height, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, imageFileName, 0);
	ImageManager::GenericWriter(tmp, image_file_name, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

#pragma endregion

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

void exercise6()
{
	//STEP 1
	load_source_image("terrain3Kx3K.tif");

	//TODO: Setup the kernel settings
	square_ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	square_ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	square_ks.dimGrid = dim3((image_width + BLOCK_DIM - 1) / BLOCK_DIM, (image_height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	//Step 2 - create heighmap texture stored in the linear pitch memory
	create_src_texure();

	//Step 3 - create the normal map
	create_normal_map();

	//Step 4 - save the normal map
	save_tex_image("normalMap.bmp");

	release_memory();
}
