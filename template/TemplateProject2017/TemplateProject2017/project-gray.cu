// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>


#include "imageKernels.cuh"

#define BLOCK_DIM 8

#define COLORS 256

texture<float, 2, cudaReadModeElementType> tex_ref;
cudaChannelFormatDesc tex_channel_desc;

unsigned char *d_image_data = nullptr;
unsigned int image_width;
unsigned int image_height;
unsigned int image_bpp;		//Bits Per Pixel = 8, 16, 24, or 32 bit
unsigned int image_pitch;

size_t histogram_tex_pitch;
size_t image_tex_pitch;
float *d_linear_pitch_texture_data = nullptr;
cudaArray *d_array_texture_data = nullptr;

uchar3 *dst_histogram_data;
uchar3 *dst_image_data;

unsigned int histogram_width = 255;
unsigned int histogram_height = 200;

KernelSetting image_ks;
KernelSetting histogram_ks;

float *d_output_data = nullptr;

const unsigned int size = COLORS * sizeof(float);

float *d_color;

int *d_max;
float *d_searching;

float h_color[COLORS];

int h_max;
float h_searching;


__global__ void search_colors(const unsigned int dst_pitch, float *d_color, float *d_searching, uchar3* dst)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * (dst_pitch / 3);
	const float texel = tex2D(tex_ref, col, row);

	//printf("%f  %f  %f\n", texel.z, texel.y, texel.x);

	uchar3 output;
	output.x = texel/3;
	output.y = texel/3;
	output.z = texel/3;

	if (texel == d_searching[0])
	{
		output.z = 255;
	}

	dst[offset] = output;

	d_color[static_cast<int>(texel)]++;
}

__global__ void draw_histogram(const unsigned int tex_width, const unsigned int tex_height, const unsigned int dst_pitch, float *d_color, int *d_max, uchar3* dst)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * (dst_pitch);

	uchar3 texel;

	//printf("%d   %d\n", row, col);

	texel.x = 0;
	texel.y = 0;
	//texel.z = (255 * col * (tex_height - row) ) / (tex_width * tex_height);
	//texel.z = (255 * col * row) / (tex_width * tex_height);
	texel.z = 0;

	const auto height = (255 * d_color[col]) / d_max[0];

	if (height < row)
	{
		texel.z = 255;
	}

	dst[offset] = texel;
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

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_image_data), image_pitch * image_height * image_bpp / 8));
	checkCudaErrors(cudaMemcpy(d_image_data, FreeImage_GetBits(tmp), image_pitch * image_height * image_bpp / 8, cudaMemcpyHostToDevice));

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
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&d_linear_pitch_texture_data), &image_tex_pitch, image_width * sizeof(float), image_height));

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch (image_bpp)
	{
	case 8:  colorToFloat<8> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 16: colorToFloat<16> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 24: colorToFloat<24> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	case 32: colorToFloat<32> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(float), d_linear_pitch_texture_data); break;
	}

	//checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "", "");

	//Texture settings
	tex_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	tex_ref.normalized = false;
	tex_ref.filterMode = cudaFilterModePoint;
	tex_ref.addressMode[0] = cudaAddressModeClamp;
	tex_ref.addressMode[1] = cudaAddressModeClamp;

	checkCudaErrors(cudaBindTexture2D(nullptr, &tex_ref, d_linear_pitch_texture_data, &tex_channel_desc, image_width, image_height, image_tex_pitch));
}
#pragma endregion

#pragma region STEP 3

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

void create_normal_map()
{
	//T ODO: Allocate Pitch memory dstTexData to store output texture
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&dst_image_data), &image_tex_pitch, image_width * sizeof(uchar3), image_height));
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&dst_histogram_data), &histogram_tex_pitch, histogram_width * sizeof(uchar3), histogram_height));


	//T ODO: Call the kernel that creates the normal map.

	for (auto i = 0; i < 256; i++)
	{
		h_color[i] = 0;
	}

	checkCudaErrors(cudaMalloc(&d_color, size));
	checkCudaErrors(cudaMalloc(&d_searching, sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_color, h_color, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_searching, &h_searching, sizeof(float), cudaMemcpyHostToDevice));

	search_colors << <image_ks.dimGrid, image_ks.dimBlock >> > (image_tex_pitch, d_color, d_searching, dst_image_data);

	checkCudaErrors(cudaMemcpy(h_color, d_color, size, cudaMemcpyDeviceToHost));
	
	h_max = 0;

	for (auto i = 0; i < 256; i++)
	{
		if (h_color[i] > h_max)
		{
			h_max = h_color[i];
		}
	}

	checkCudaErrors(cudaMalloc(&d_max, sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice));

	draw_histogram << <histogram_ks.dimGrid, histogram_ks.dimBlock >> > (histogram_width, histogram_height, histogram_tex_pitch, d_color, d_max, dst_histogram_data);

	double count = 0;

	for (auto i = 0; i < 256; i++)
	{
		count += h_color[i];
		printf("%d: %f\n", i, h_color[i]);
	}
	printf("total: %f, %d\n", count, image_width * image_height);
	//check_data<uchar3>::checkDeviceMatrix(dstTexData, image_height, texPitch / sizeof(uchar3), true, "%hhu %hhu %hhu | ", "Result of Linear Pitch Text");
}

#pragma endregion

#pragma region STEP 4

//TASK: Save output image (normal map)

void save_histogram_image(const char* image_file_name)
{
	FreeImage_Initialise();

	const auto tmp = FreeImage_Allocate(histogram_width, histogram_height, 24);
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dst_histogram_data, histogram_tex_pitch, histogram_width * 3, histogram_height, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, image_file_name, 0);
	ImageManager::GenericWriter(tmp, image_file_name, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
void save_image(const char* image_file_name)
{
	FreeImage_Initialise();

	const auto tmp = FreeImage_Allocate(image_width, image_height, 24);
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dst_image_data, image_tex_pitch, image_width * 3, image_height, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, image_file_name, 0);
	ImageManager::GenericWriter(tmp, image_file_name, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

#pragma endregion

void release_memory()
{
	checkCudaErrors(cudaFree(d_color));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_searching));
	
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

void project_gray()
{
	//STEP 1
	h_searching = 100;

	load_source_image("terrain3Kx3K.tif");

	//TODO: Setup the kernel settings
	image_ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	image_ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	image_ks.dimGrid = dim3((image_width + BLOCK_DIM - 1) / BLOCK_DIM, (image_height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	histogram_ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	histogram_ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	histogram_ks.dimGrid = dim3((histogram_width + BLOCK_DIM - 1) / BLOCK_DIM, (histogram_height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	//Step 2 - create heighmap texture stored in the linear pitch memory
	create_src_texure();

	//Step 3 - create the normal map
	create_normal_map();

	//Step 4 - save the normal map
	save_histogram_image("histogram.bmp");
	save_image("founded.bmp");

	printf("Success\n");

	release_memory();
}
