#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <glew.h>
#include <freeglut.h>
#include <cudaDefs.h>
#include <imageManager.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_gl.h>      // helper functions for CUDA/GL interop

#define BLOCK_DIM 8

#define COLORS 256

//cudaError_t error = cudaSuccess;
//cudaDeviceProp deviceProp = cudaDeviceProp();

//CUDA variables
unsigned int image_width;
unsigned int image_height;
cudaGraphicsResource_t cuda_pbo_resource;
cudaGraphicsResource_t cuda_tex_resource;
texture<uchar4, 2, cudaReadModeElementType> cuda_tex_ref;
cudaChannelFormatDesc cuda_tex_channel_desc;
KernelSetting ks;
unsigned char value = 0;

//backup

uchar4 *d_backup;
size_t backup_pitch;

//OpenGL
unsigned int pbo_id;
unsigned int texture_id;

unsigned int viewport_width = 1024;
unsigned int viewport_height = 1024;

const int size = COLORS * sizeof(float);

float *d_red;
float *d_green;
float *d_blue;

float *d_max;
float *d_searching;

float *h_red;
float *h_green;
float *h_blue;

float h_max[4];
float h_searching[6];

int h_mouse_click[2];
int *d_mouse_click;

int h_viewport_size[2];
int *d_viewport_size;

int h_image_size[2];
int *d_image_size;

bool h_is_selected = false;

#pragma region CUDA Routines

__global__ void apply_filter(int value, int *d_image_size, float *d_red, float *d_green, float *d_blue, int *d_mouse_click, float *d_searching, unsigned char *pbo)
{

	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * d_image_size[0];
	uchar4 texel = tex2D(cuda_tex_ref, col, row);

	if (d_mouse_click[0] != -1)
	{
		if (texel.y == 255 && texel.z == 255)
		{
			texel.x = value;
		}
	}

	const auto uchar4_pbo = reinterpret_cast<uchar4*>(pbo);

	uchar4_pbo[offset] = texel;
}

__global__ void apply_filter_first_run(int *d_image_size, float *d_red, float *d_green, float *d_blue, uchar4* backup, unsigned int pitch, unsigned char *pbo)
{

	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * d_image_size[0];
	const uchar4 texel = tex2D(cuda_tex_ref, col, row);

	d_red[texel.x]++;
	d_green[texel.y]++;
	d_blue[texel.z]++;

	const auto offset_backup = col + row * (pitch / 4);
	backup[offset_backup] = texel;

	const auto uchar4_pbo = reinterpret_cast<uchar4*>(pbo);
	uchar4_pbo[offset] = texel;
}

__global__ void apply_filter_restore(int *d_image_size, uchar4* backup, const unsigned int pitch, unsigned char *pbo)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * d_image_size[0];
	const auto offset_backup = col + row * (pitch / 4);

	const auto uchar4_pbo = reinterpret_cast<uchar4*>(pbo);
	uchar4_pbo[offset] = backup[offset_backup];
}

__global__ void apply_filter_click(int *d_image_size, float *d_searching, uchar4* backup, const unsigned int pitch, unsigned char *pbo)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * d_image_size[0];
	const auto offset_backup = col + row * (pitch / 4);
	auto texel = backup[offset_backup];

	if (texel.x >= d_searching[0] && texel.y >= d_searching[1] && texel.z >= d_searching[2] && texel.x <= d_searching[3] && texel.y <= d_searching[4] && texel.z <= d_searching[5])
	{
		texel.x = 255;
		texel.y = 255;
		texel.z = 255;
	}
	else
	{
		texel.x = texel.x >> 3;
		texel.y = texel.y >> 3;
		texel.z = texel.z >> 3;
	}

	const auto uchar4_pbo = reinterpret_cast<uchar4*>(pbo);

	uchar4_pbo[offset] = texel;
}

__device__ bool check_searched(int value, uchar4 &to_test, uchar4 &result)
{
	if (to_test.y == 255)
	{
		result.x = value;
		result.y = 255;
		result.z = 255;

		return false;
	}
	return true;
}

__global__ void search_neighbourhood (int value, int *d_image_size, unsigned char *pbo)
{

	auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * d_image_size[0];
	uchar4 texel = tex2D(cuda_tex_ref, col, row);
	uchar4 s = tex2D(cuda_tex_ref, col, row + 1);
	uchar4 es = tex2D(cuda_tex_ref, col + 1, row + 1);
	uchar4 e = tex2D(cuda_tex_ref, col + 1, row);
	uchar4 en = tex2D(cuda_tex_ref, col + 1, row - 1);
	uchar4 n = tex2D(cuda_tex_ref, col, row - 1);
	uchar4 nw = tex2D(cuda_tex_ref, col - 1, row + 1);
	uchar4 w = tex2D(cuda_tex_ref, col - 1, row);
	uchar4 sw = tex2D(cuda_tex_ref, col - 1, row - 1);

	if (check_searched(value, s, texel))
	{
		if (check_searched(value, es, texel))
		{
			if (check_searched(value, e, texel))
			{
				if (check_searched(value, en, texel))
				{
					if (check_searched(value, n, texel))
					{
						if (check_searched(value, nw, texel))
						{
							if (check_searched(value, w, texel))
							{
								check_searched(value, sw, texel);
							}
						}
					}
				}
			}
		}
	}
	
	const auto uchar4_pbo = reinterpret_cast<uchar4*>(pbo);

	uchar4_pbo[offset] = texel;
}

__global__ void search_color(float *d_searching, int *d_mouse_click, int *d_viewport_size, int *d_image_size)
{
	printf("Clicked (viewport): %d %d\n", d_mouse_click[0], d_mouse_click[1]);

	auto mouse_x = (d_mouse_click[0] * d_image_size[0]) / d_viewport_size[0];
	auto mouse_y = d_image_size[1] - (d_mouse_click[1] * d_image_size[1]) / d_viewport_size[1];

	printf("Clicked (image): %d %d\n", mouse_x, d_image_size[1] - mouse_y);

	// ReSharper disable once CppLocalVariableMayBeConst
	uchar4 texel = tex2D(cuda_tex_ref, mouse_x, mouse_y);

	d_searching[0] = texel.x;
	d_searching[1] = texel.y;
	d_searching[2] = texel.z;

	d_searching[3] = texel.x;
	d_searching[4] = texel.y;
	d_searching[5] = texel.z;

	printf("Selected color: %d %d %d\n", texel.x, texel.y, texel.z);
}

__global__ void draw_histogram(const unsigned int histogram_height, const unsigned int dst_pitch, float *d_red, float *d_green, float *d_blue, float *d_max, uchar4* dst)
{
	const auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	const auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset4 = col + row * (dst_pitch / 4);
	const auto offset3 = col + row * (dst_pitch / 4) + histogram_height * (dst_pitch / 4);
	const auto offset2 = col + row * (dst_pitch / 4) + histogram_height * (dst_pitch / 4) * 2;
	const auto offset1 = col + row * (dst_pitch / 4) + histogram_height * (dst_pitch / 4) * 3;
	uchar4 texel1;

	texel1.w = 255;
	texel1.x = 0;
	texel1.y = 0;
	texel1.z = 0;

	auto texel2 = texel1;
	auto texel3 = texel1;
	auto texel4 = texel1;

	const auto height_red = (histogram_height * d_red[col]) / d_max[0];
	if (height_red > row)
	{
		texel2.z = 255;
	}

	const auto height_green = (histogram_height * d_green[col]) / d_max[1];
	if (height_green > row)
	{
		texel3.y = 255;
	}

	const auto height_blue = (histogram_height * d_blue[col]) / d_max[2];
	if (height_blue > row)
	{
		texel4.x = 255;
	}

	const auto height_red_rgb = (histogram_height * d_red[col]) / d_max[3];
	if (height_red_rgb > row)
	{
		texel1.z = 255;
	}

	const auto height_green_rgb = (histogram_height * d_green[col]) / d_max[3];
	if (height_green_rgb > row)
	{
		texel1.y = 255;
	}

	const auto height_blue_rgb = (histogram_height * d_blue[col]) / d_max[3];
	if (height_blue_rgb > row)
	{
		texel1.x = 255;
	}

	dst[offset1] = texel1;
	dst[offset2] = texel2;
	dst[offset3] = texel3;
	dst[offset4] = texel4;
}

void mouse_click(const int button, const int state, const int x, const int y)
{

	if (!h_is_selected && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		h_mouse_click[0] = x;
		h_mouse_click[1] = y;

		cudaArray* array;
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, nullptr));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_resource, 0, 0));
		checkCudaErrors(cudaGetChannelDesc(&cuda_tex_channel_desc, array));
		checkCudaErrors(cudaBindTextureToArray(&cuda_tex_ref, array, &cuda_tex_channel_desc));

		unsigned char *pbo_data;
		size_t pbo_size;
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));

		checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_data), &pbo_size, cuda_pbo_resource));

		checkCudaErrors(cudaMemcpy(d_mouse_click, h_mouse_click, sizeof(int) * 2, cudaMemcpyHostToDevice));

		search_color << < 1, 1 >> > (d_searching, d_mouse_click, d_viewport_size, d_image_size);

		apply_filter_click << <ks.dimGrid, ks.dimBlock >> > (d_image_size, d_searching, d_backup, backup_pitch, pbo_data);

		checkCudaErrors(cudaUnbindTexture(&cuda_tex_ref));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, nullptr));

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

		h_is_selected = true;
	}
}

void save_histogram(const char *filename)
{
	uchar4 *d_result;
	size_t result_pitch;

	const auto histogram_width = 255;
	const auto histogram_height = 200;

	checkCudaErrors(cudaMallocPitch(&d_result, &result_pitch, histogram_width * 4, histogram_height * 4));

	KernelSetting histogram_ks;

	histogram_ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	histogram_ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	histogram_ks.dimGrid = dim3((histogram_width + BLOCK_DIM - 1) / BLOCK_DIM, (histogram_height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

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
	h_max[3] = h_max[0];

	if (h_max[1] > h_max[3])
		h_max[3] = h_max[1];

	if (h_max[2] > h_max[3])
		h_max[3] = h_max[2];

	checkCudaErrors(cudaMalloc(&d_max, 4 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_max, h_max, 4 * sizeof(float), cudaMemcpyHostToDevice));

	draw_histogram << <histogram_ks.dimGrid, histogram_ks.dimBlock >> > (histogram_height, result_pitch, d_red, d_green, d_blue, d_max, d_result);

	// ReSharper disable once CppUseAuto
	FIBITMAP *h_image = FreeImage_Allocate(histogram_width, histogram_height * 4, 32);

	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(h_image), FreeImage_GetPitch(h_image), d_result, result_pitch, histogram_width * 4, histogram_height * 4, cudaMemcpyDeviceToHost));

	FreeImage_Save(FIF_PNG, h_image, filename, 0);

	FreeImage_Unload(h_image);

	checkCudaErrors(cudaFree(d_result));

	printf("Saved histogram as %s\n", filename);
}

void make_founds_bigger()
{
	if (!h_is_selected)
	{
		return;
	}
	
	cudaArray* array;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, nullptr));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_resource, 0, 0));
	checkCudaErrors(cudaGetChannelDesc(&cuda_tex_channel_desc, array));
	checkCudaErrors(cudaBindTextureToArray(&cuda_tex_ref, array, &cuda_tex_channel_desc));

	unsigned char *pbo_data;
	size_t pbo_size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_data), &pbo_size, cuda_pbo_resource));

	search_neighbourhood << <ks.dimGrid, ks.dimBlock >> > (value, d_image_size, pbo_data);

	checkCudaErrors(cudaUnbindTexture(&cuda_tex_ref));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, nullptr));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

void restore_image()
{
	cudaArray* array;

	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, nullptr));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_resource, 0, 0));
	checkCudaErrors(cudaGetChannelDesc(&cuda_tex_channel_desc, array));
	checkCudaErrors(cudaBindTextureToArray(&cuda_tex_ref, array, &cuda_tex_channel_desc));

	unsigned char *pbo_data;
	size_t pbo_size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_data), &pbo_size, cuda_pbo_resource));

	apply_filter_restore<< <ks.dimGrid, ks.dimBlock >> > (d_image_size, d_backup, backup_pitch, pbo_data);

	checkCudaErrors(cudaUnbindTexture(&cuda_tex_ref));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, nullptr));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);   //Source parameter is NULL, Data is coming from a PBO, not host memory

	h_mouse_click[0] = h_mouse_click[1] = -1;
	checkCudaErrors(cudaMemcpy(d_mouse_click, h_mouse_click, sizeof(int) * 2, cudaMemcpyHostToDevice));

	h_is_selected = false;

	printf("Restored image\n\n");
}

void change_searching_range()
{
	if (!h_is_selected)
	{
		return;
	}
	
	checkCudaErrors(cudaMemcpy(h_searching, d_searching, 6 * sizeof(float), cudaMemcpyDeviceToHost));
	if (h_searching[0] >= 1)
	{
		h_searching[0]--;
	}
	if (h_searching[1] >= 1)
	{
		h_searching[1]--;
	}
	if (h_searching[2] >= 1)
	{
		h_searching[2]--;
	}

	if (h_searching[3] <= 254)
	{
		h_searching[3]++;
	}
	if (h_searching[4] <= 254)
	{
		h_searching[4]++;
	}
	if (h_searching[5] <= 254)
	{
		h_searching[5]++;
	}
	checkCudaErrors(cudaMemcpy(d_searching, h_searching, 6 * sizeof(float), cudaMemcpyHostToDevice));

	cudaArray* array;

	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, nullptr));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_resource, 0, 0));
	checkCudaErrors(cudaGetChannelDesc(&cuda_tex_channel_desc, array));
	checkCudaErrors(cudaBindTextureToArray(&cuda_tex_ref, array, &cuda_tex_channel_desc));

	unsigned char *pbo_data;
	size_t pbo_size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_data), &pbo_size, cuda_pbo_resource));

	apply_filter_click << <ks.dimGrid, ks.dimBlock >> > (d_image_size, d_searching, d_backup, backup_pitch, pbo_data);

	checkCudaErrors(cudaUnbindTexture(&cuda_tex_ref));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, nullptr));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);   //Source parameter is NULL, Data is coming from a PBO, not host memory

	printf("Changed range of searched color\n");
}

void keyboard(const unsigned char key, int x, int y)
{
	if (key == 27)
	{
		exit(EXIT_SUCCESS);
	}

	if (key == 's')
	{
		save_histogram("histogram.png");
	}
	
	if (key == 'm')
	{
		make_founds_bigger();
	}
	
	if (key == 'r')
	{
		restore_image();
	}

	if (key == 'c')
	{
		change_searching_range();
	}
}

void cuda_worker_first_run()
{
	cudaArray* array;

	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, nullptr));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_resource, 0, 0));
	checkCudaErrors(cudaGetChannelDesc(&cuda_tex_channel_desc, array));
	checkCudaErrors(cudaBindTextureToArray(&cuda_tex_ref, array, &cuda_tex_channel_desc));

	unsigned char *pbo_data;
	size_t pbo_size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_data), &pbo_size, cuda_pbo_resource));

	checkCudaErrors(cudaMallocPitch(&d_backup, &backup_pitch, image_width * 4, image_height));

	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.dimGrid = dim3((image_width + BLOCK_DIM - 1) / BLOCK_DIM, (image_height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	apply_filter_first_run << <ks.dimGrid, ks.dimBlock >> > (d_image_size, d_red, d_green, d_blue, d_backup, backup_pitch, pbo_data);

	checkCudaErrors(cudaUnbindTexture(&cuda_tex_ref));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, nullptr));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);   //Source parameter is NULL, Data is coming from a PBO, not host memory

	checkCudaErrors(cudaMemcpy(h_red, d_red, size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_green, d_green, size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_blue, d_blue, size, cudaMemcpyDeviceToHost));

	h_mouse_click[0] = -1;
	checkCudaErrors(cudaMemcpy(d_mouse_click, h_mouse_click, sizeof(int) * 2, cudaMemcpyHostToDevice));
}

void cuda_worker()
{
	cudaArray* array;

	//T ODO 3: Map cudaTexResource
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, nullptr));

	//T ODO 4: Get Mapped Array of cudaTexResource
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_resource, 0, 0));

	//T ODO 5: Get cudaTexChannelDesc from previously obtained array
	checkCudaErrors(cudaGetChannelDesc(&cuda_tex_channel_desc, array));

	//T ODO 6: Bind cudaTexRef to array
	checkCudaErrors(cudaBindTextureToArray(&cuda_tex_ref, array, &cuda_tex_channel_desc));

	unsigned char *pbo_data;
	size_t pbo_size;
	//T ODO 7: Map cudaPBOResource
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));

	//T ODO 7: Map Mapped pointer to cudaPBOResource data
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_data), &pbo_size, cuda_pbo_resource));

	//T ODO 8: Set KernelSetting variable ks (dimBlock, dimGrid, etc.) such that block will have BLOCK_DIM x BLOCK_DIM threads
	//done in cuda_worker_first_run

	//Calling applyFilter kernel
	value-=2;
	if (value < 10) value = 255;

	apply_filter << <ks.dimGrid, ks.dimBlock >> > (value, d_image_size, d_red, d_green, d_blue, d_mouse_click, d_searching, pbo_data);

	//Following code release mapped resources, unbinds texture and ensures that PBO data will be copied into OpenGL texture. Do not modify following code!
	checkCudaErrors(cudaUnbindTexture(&cuda_tex_ref));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, nullptr));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);   //Source parameter is NULL, Data is coming from a PBO, not host memory
}

void init_cuda_tex()
{
	// ReSharper disable once CppDeprecatedEntity
	cudaGLSetGLDevice(0);
	checkError();

	//T ODO 1: Register OpenGL texture to CUDA resource
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

	//CUDA Texture settings
	cuda_tex_ref.normalized = false;						//Otherwise TRUE to access with normalized texture coordinates
	cuda_tex_ref.filterMode = cudaFilterModePoint;			//Otherwise texRef.filterMode = cudaFilterModeLinear; for Linear interpolation of texels
	cuda_tex_ref.addressMode[0] = cudaAddressModeClamp;		//No repeat texture pattern
	cuda_tex_ref.addressMode[1] = cudaAddressModeClamp;		//No repeat texture pattern

	//T ODO 2: Register PBO to CUDA resource
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_id, cudaGraphicsRegisterFlagsWriteDiscard));
}

void release_cuda()
{
	checkCudaErrors(cudaFree(d_backup));
	
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_searching));
	checkCudaErrors(cudaFree(d_mouse_click));
	checkCudaErrors(cudaFree(d_viewport_size));
	checkCudaErrors(cudaFree(d_image_size));
	
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_resource));
}
#pragma endregion

#pragma region OpenGL Routines - DO NOT MODIFY THIS SECTION !!!

void load_texture(const char* image_file_name)
{
	FreeImage_Initialise();
	const auto temp = ImageManager::GenericLoader(image_file_name, 0);

	image_width = FreeImage_GetWidth(temp);
	image_height = FreeImage_GetHeight(temp);

	//OpenGL Texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);

	//WARNING: Just some of inner format are supported by CUDA!!!
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits(temp));
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	FreeImage_Unload(temp);

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
	checkCudaErrors(cudaMalloc(&d_searching, 6 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_red, h_red, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_green, h_green, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_blue, h_blue, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_searching, h_searching, 6 * sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_mouse_click, sizeof(int) * 2));

	h_mouse_click[0] = h_mouse_click[1] = -1;
	checkCudaErrors(cudaMemcpy(d_mouse_click, h_mouse_click, sizeof(int) * 2, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_viewport_size, sizeof(int) * 2));
	h_viewport_size[0] = viewport_width;
	h_viewport_size[1] = viewport_height;
	checkCudaErrors(cudaMemcpy(d_viewport_size, h_viewport_size, sizeof(int) * 2, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_image_size, sizeof(int) * 2));
	h_image_size[0] = image_width;
	h_image_size[1] = image_height;
	checkCudaErrors(cudaMemcpy(d_image_size, h_image_size, sizeof(int) * 2, cudaMemcpyHostToDevice));
}

void prepare_pbo()
{
	glGenBuffers(1, &pbo_id);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);												// Make this the current UNPACK buffer (OpenGL is state-based)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, image_width * image_height * 4, nullptr, GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
}

void my_display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_id);

	//I know this is a very old OpenGL, but we want to practice CUDA :-)
		//Now it will be a wasted time to learn you current features of OpenGL. Sorry for that however, you can visit my second seminar dealing with Computer Graphics (CG2).
	glBegin(GL_QUADS);

	glTexCoord2d(0, 0);		glVertex2d(0, 0);
	glTexCoord2d(1, 0);		glVertex2d(viewport_width, 0);
	glTexCoord2d(1, 1);		glVertex2d(viewport_width, viewport_height);
	glTexCoord2d(0, 1);		glVertex2d(0, viewport_height);

	glEnd();

	glDisable(GL_TEXTURE_2D);

	glFlush();
	glutSwapBuffers();
}

void my_resize(const GLsizei w, const GLsizei h)
{
	viewport_width = w;
	viewport_height = h;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, viewport_width, viewport_height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, viewport_width, 0, viewport_height);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();

	h_viewport_size[0] = w;
	h_viewport_size[1] = h;
	checkCudaErrors(cudaMemcpy(d_viewport_size, h_viewport_size, sizeof(int) * 2, cudaMemcpyHostToDevice));
}

void my_idle()
{
	cuda_worker();
	glutPostRedisplay();
}

void init_gl(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(viewport_width, viewport_height);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Image");

	glutDisplayFunc(my_display);
	glutReshapeFunc(my_resize);
	glutIdleFunc(my_idle);
	glutMouseFunc(mouse_click);
	glutKeyboardFunc(keyboard);
	glutSetCursor(GLUT_CURSOR_CROSSHAIR);

	// initialize necessary OpenGL extensions
	glewInit();

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glShadeModel(GL_SMOOTH);
	glViewport(0, 0, viewport_width, viewport_height);

	glFlush();
}

void release_open_gl()
{
	if (texture_id > 0)
		glDeleteTextures(1, &texture_id);
	if (pbo_id > 0)
		glDeleteBuffers(1, &pbo_id);
}

#pragma endregion

void release_resources()
{
	release_cuda();
	release_open_gl();
}

void project(const int argc, char *argv[])
{
	//initializeCUDA(deviceProp);

	init_gl(argc, argv);

	load_texture("lena.png");
	//load_texture("image.tif");
	//load_texture("testing.png");

	prepare_pbo();

	init_cuda_tex();

	cuda_worker_first_run();

	printf("------------------------------------------------------------------------\n\n");
	printf("To select color, use the cursor and click somewhere on the image\n");
	printf("Keyboard shortcuts: \n\t s - create and Save histogram (histogram.png) \n\t m - Make founds (if exist) bigger \n\t r - Restore image (hide founds and make image brighter) \n\t c - Change range of searched color by +-1 \n\t ESC - close image and terminate program\n\n");
	printf("------------------------------------------------------------------------\n");

	//start rendering main loop
	glutMainLoop();
	atexit(release_resources);
}
