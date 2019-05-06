// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>

#include "imageKernels.cuh"

#define BLOCK_DIM 8

#define COLORS 256

texture<uchar4, 2, cudaReadModeElementType> tex_ref;
cudaChannelFormatDesc tex_channel_desc;

uchar4 *d_image_data;
unsigned int image_width;
unsigned int image_height;
unsigned int image_bpp;		//Bits Per Pixel = 8, 16, 24, or 32 bit
size_t image_pitch;

size_t histogram_tex_pitch;
size_t image_tex_pitch;
uchar4 *d_image_linear_pitch_texture_data = nullptr;
cudaArray *d_array_texture_data = nullptr;

uchar3 *dst_histogram_data;
uchar3 *dst_image_data;

KernelSetting image_ks;
KernelSetting histogram_ks;

float *d_output_data = nullptr;

unsigned int histogram_width = 255;
unsigned int histogram_height = 200;

const int size = COLORS * sizeof(float);

float *d_red;
float *d_green;
float *d_blue;

float *d_max;
float *d_color;

float *h_red;
float *h_green;
float *h_blue;

float h_max[3];
float h_color[3];


__global__ void search_colors(const unsigned int tex_width, const unsigned int tex_height, const unsigned int dst_pitch, float *d_red, float *d_green, float *d_blue, float *d_color, uchar3* dst)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * (dst_pitch / 3);
	const uchar4 texel = tex2D(tex_ref, col, row);

	//printf("%f  %f  %f\n", texel.z, texel.y, texel.x);

	uchar3 output;
	output.x = texel.z;
	output.y = texel.y;
	output.z = texel.x;

	dst[offset] = output;

	d_red[texel.x]++;
	d_green[texel.y]++;
	d_blue[texel.z]++;
}

__global__ void draw_histogram (const unsigned int tex_width, const unsigned int tex_height, const unsigned int dst_pitch, float *d_red, float *d_green, float *d_blue, float *d_max, uchar3* dst)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * (dst_pitch / 1);
	uchar3 texel;

	/*[texel.x]++;
	d_green[texel.y]++;
	d_blue[texel.z]++;*/

	//printf("%d   %d\n", col, row);

	texel.x = 100;
	texel.y = 0;
	texel.z = 0;

	dst[offset] = texel;
}

#pragma region STEP 1


void load_source_image(const char* image_file_name)
{
	FreeImage_Initialise();
	auto tmp = ImageManager::GenericLoader(image_file_name, 0);
	tmp = FreeImage_ConvertTo32Bits(tmp);

	image_width = FreeImage_GetWidth(tmp);
	image_height = FreeImage_GetHeight(tmp);
	image_bpp = FreeImage_GetBPP(tmp);
	image_pitch = FreeImage_GetPitch(tmp);

	//checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_image_data), image_pitch * image_height * image_bpp / 8));
	//checkCudaErrors(cudaMemcpy(d_image_data, FreeImage_GetBits(tmp), image_pitch * image_height * image_bpp / 8, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMallocPitch(&d_image_data, &image_pitch, image_width * sizeof(uchar4), image_height));
	checkCudaErrors(cudaMemcpy2D(d_image_data, image_pitch, FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), image_width * sizeof(uchar4), image_height, cudaMemcpyHostToDevice));

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
#pragma endregion

#pragma region STEP 2


void create_src_texure()
{
	//Floating Point Texture Data
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&d_image_linear_pitch_texture_data), &image_tex_pitch, image_width * sizeof(uchar4), image_height));

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	/*switch (image_bpp)
	{
	case 8:  colorToUchar4<8> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(uchar4), d_image_linear_pitch_texture_data); break;
	case 16: colorToUchar4<16> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(uchar4), d_image_linear_pitch_texture_data); break;
	case 24: colorToUchar4<24> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(uchar4), d_image_linear_pitch_texture_data); break;
	case 32: colorToUchar4<32> << <image_ks.dimGrid, image_ks.dimBlock >> > (d_image_data, image_width, image_height, image_pitch, image_tex_pitch / sizeof(uchar4), d_image_linear_pitch_texture_data); break;
	}*/

	//checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "", "");

	//Texture settings
	//tex_channel_desc = cudaCreateChannelDesc<uchar4>();
	tex_channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	tex_ref.normalized = false;
	tex_ref.filterMode = cudaFilterModePoint;
	tex_ref.addressMode[0] = cudaAddressModeClamp;
	tex_ref.addressMode[1] = cudaAddressModeClamp;

	checkCudaErrors(cudaBindTexture2D(nullptr, &tex_ref, d_image_data, &tex_channel_desc, image_width, image_height, image_tex_pitch));
}
#pragma endregion

#pragma region STEP 3

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

void create_histogram()
{
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&dst_image_data), &image_tex_pitch, image_width * sizeof(uchar3), image_height));

	h_red = static_cast<float*>(malloc(size));
	h_green = static_cast<float*>(malloc(size));
	h_blue = static_cast<float*>(malloc(size));

	for (auto i = 0; i < COLORS; i++)
	{
		h_red[i] = 0;
		h_green[i] = 0;
		h_blue[i] = 0;
	}

	checkCudaErrors(cudaMalloc(&d_red, size));
	checkCudaErrors(cudaMalloc(&d_green, size));
	checkCudaErrors(cudaMalloc(&d_blue, size));
	checkCudaErrors(cudaMalloc(&d_color, 3 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_red, h_red, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_green, h_green, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_blue, h_blue, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_color, h_color, 3 * sizeof(float), cudaMemcpyHostToDevice));

	search_colors << <image_ks.dimGrid, image_ks.dimBlock >> > (image_width, image_height, image_tex_pitch, d_red, d_green, d_blue, d_color, dst_image_data);

	checkCudaErrors(cudaMemcpy(h_red, d_red, size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_green, d_green, size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_blue, d_blue, size, cudaMemcpyDeviceToHost));

	
	for (auto i = 0; i < 256; i++)
	{
		if (h_red[i] > h_max[0])
		{
			h_max[0] = h_red[i];
		}

		if (h_green[i] > h_max[1])
		{
			h_max[1] = h_green[i];
		}

		if (h_blue[i] > h_max[2])
		{
			h_max[2] = h_blue[i];
		}
	}

	checkCudaErrors(cudaMalloc(&d_max, 3 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_max, h_max, 3 * sizeof(float), cudaMemcpyHostToDevice));

	draw_histogram << <histogram_ks.dimGrid, histogram_ks.dimBlock >> > (histogram_width, histogram_height, histogram_tex_pitch, d_red, d_green, d_blue, d_max, dst_histogram_data);
}

#pragma endregion

#pragma region STEP 4

//TASK: Save output image (normal map)

void save_histogram_image(const char* image_file_name)
{
	FreeImage_Initialise();

	const auto tmp = FreeImage_Allocate(histogram_width, histogram_height, 24);
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dst_histogram_data, histogram_tex_pitch, histogram_width * 3, histogram_height, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, imageFileName, 0);
	ImageManager::GenericWriter(tmp, image_file_name, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
void save_image(const char* image_file_name)
{
	FreeImage_Initialise();

	const auto tmp = FreeImage_Allocate(image_width, image_height, 32);
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dst_image_data, image_tex_pitch, image_width * 4, image_height, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, image_file_name, 0);
	FreeImage_Save(FIF_PNG, tmp, image_file_name, 0);
	//ImageManager::GenericWriter(tmp, image_file_name, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

#pragma endregion

void release_memory()
{
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_color));
	
	free(h_red);
	free(h_green);
	free(h_blue);

	h_red = nullptr;
	h_green = nullptr;
	h_blue = nullptr;
	
	cudaUnbindTexture(tex_ref);
	if (d_image_data != nullptr)
		checkCudaErrors(cudaFree(d_image_data));
	if (d_image_linear_pitch_texture_data != nullptr)
		checkCudaErrors(cudaFree(d_image_linear_pitch_texture_data));
	if (d_array_texture_data)
		checkCudaErrors(cudaFreeArray(d_array_texture_data));
	if (d_output_data)
		checkCudaErrors(cudaFree(d_output_data));
}

void project_color()
{
	//STEP 1
	load_source_image("image.tif");

	h_color[0] = 100;
	h_color[1] = 100;
	h_color[2] = 100;

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
	create_histogram();

	//Step 4 - save the normal map
	//save_histogram_image("histogram.bmp");
	save_image("founded.bmp");

	release_memory();

	std::getchar();
}
