#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_gl.h>      // helper functions for CUDA/GL interop

#include "imageKernels.cuh"

#define BLOCK_DIM 8

//cudaError_t error = cudaSuccess;
//cudaDeviceProp deviceProp = cudaDeviceProp();

//CUDA variables
unsigned int imageWidth;
unsigned int imageHeight;
unsigned int imageBPP;		//Bits Per Pixel = 8, 16, 24, or 32 bit
unsigned int imagePitch;
cudaGraphicsResource_t cuda_pbo_resource;
cudaGraphicsResource_t cuda_tex_resource;
texture<uchar4, 2, cudaReadModeElementType> cuda_tex_ref;
cudaChannelFormatDesc cuda_tex_channel_desc;
KernelSetting ks;
unsigned char some_value = 0;

//OpenGL
unsigned int pbo_id;
unsigned int texture_id;

unsigned int viewport_width = 1024;
unsigned int viewport_height = 1024;

#pragma region CUDA Routines

__global__ void apply_filter(const unsigned char some_value, const unsigned int pbo_width, const unsigned int pbo_height, unsigned char *pbo)
{
	//TODO 9: Create a filter that replaces Red spectrum of RGBA pbo such that RED=someValue 
	//
	auto col = (threadIdx.x + blockIdx.x * blockDim.x);
	auto row = (threadIdx.y + blockIdx.y * blockDim.y);

	const auto offset = col + row * pbo_width;
	uchar4 texel = tex2D(cuda_tex_ref, col, row);

	texel.x = some_value;
	const auto uchar4_pbo = reinterpret_cast<uchar4*>(pbo);

	uchar4_pbo[offset] = texel;
}

void cuda_worker()
{
	cudaArray* array;

	//T ODO 3: Map cudaTexResource
	cudaGraphicsMapResources(1, &cuda_tex_resource, nullptr);
	
	//T ODO 4: Get Mapped Array of cudaTexResource
	cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_resource, 0, 0);
	
	//T ODO 5: Get cudaTexChannelDesc from previously obtained array
	cudaGetChannelDesc(&cuda_tex_channel_desc, array);	

	//T ODO 6: Binf cudaTexRef to array
	cudaBindTextureToArray(&cuda_tex_ref, array, &cuda_tex_channel_desc);
	checkError();



	unsigned char *pbo_data;
	size_t pboSize;
	//T ODO 7: Map cudaPBOResource
	cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr);
	
	//T ODO 7: Map Mapped pointer to cudaPBOResource data
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pbo_data), &pboSize, cuda_pbo_resource);
	checkError();
		
	//T ODO 8: Set KernelSetting variable ks (dimBlock, dimGrid, etc.) such that block will have BLOCK_DIM x BLOCK_DIM threads
	//...
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.dimGrid = dim3((imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);


	//Calling applyFileter kernel
	some_value++;
	if (some_value>255) some_value = 0;
	apply_filter<<<ks.dimGrid, ks.dimBlock>>>(some_value, imageWidth, imageHeight, pbo_data);

	//Following code release mapped resources, unbinds texture and ensures that PBO data will be coppied into OpenGL texture. Do not modify following code!
	cudaUnbindTexture(&cuda_tex_ref);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr);
	cudaGraphicsUnmapResources(1, &cuda_tex_resource, nullptr);
	
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo_id);
	glBindTexture( GL_TEXTURE_2D, texture_id);
	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
}

void init_cud_atex()
{
	cudaGLSetGLDevice(0);
	checkError();

	//T ODO 1: Register OpenGL texture to CUDA resource
	cudaGraphicsGLRegisterImage(&cuda_tex_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);

	//CUDA Texture settings
	cuda_tex_ref.normalized = false;						//Otherwise TRUE to access with normalized texture coordinates
	cuda_tex_ref.filterMode = cudaFilterModePoint;		//Otherwise texRef.filterMode = cudaFilterModeLinear; for Linear interpolation of texels
	cuda_tex_ref.addressMode[0] = cudaAddressModeClamp;	//No repeat texture pattern
	cuda_tex_ref.addressMode[1] = cudaAddressModeClamp;	//No repeat texture pattern

	checkError();

	//T ODO 2: Register PBO to CUDA resource
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_id, cudaGraphicsRegisterFlagsWriteDiscard);

	checkError();
}

void release_cuda()
{
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
	cudaGraphicsUnregisterResource(cuda_tex_resource);
}
#pragma endregion

#pragma region OpenGL Routines - DO NOT MODIFY THIS SECTION !!!

void load_texture(const char* image_file_name)
{
	FreeImage_Initialise();
	const auto tmp = ImageManager::GenericLoader(image_file_name, 0);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp);

	//OpenGL Texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1,&texture_id);
	glBindTexture( GL_TEXTURE_2D, texture_id);

	//WARNING: Just some of inner format are supported by CUDA!!!
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	FreeImage_Unload(tmp);
}

void prepare_pbo()
{
	glGenBuffers(1, &pbo_id);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);												// Make this the current UNPACK buffer (OpenGL is state-based)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 4, nullptr, GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
}

void my_display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_id);

	//I know this is a very old OpenGL, but we want to practice CUDA :-)
        //Now it will be a wasted time to learn you current features of OpenGL. Sorry for that however, you can visit my second seminar dealing with Computer Graphics (CG2).
	glBegin(GL_QUADS);

	glTexCoord2d(0,0);		glVertex2d(0,0);
	glTexCoord2d(1,0);		glVertex2d(viewport_width, 0);
	glTexCoord2d(1,1);		glVertex2d(viewport_width, viewport_height);
	glTexCoord2d(0,1);		glVertex2d(0, viewport_height);

	glEnd();

	glDisable(GL_TEXTURE_2D);

	glFlush();			
	glutSwapBuffers();
}

void my_resize(GLsizei w, GLsizei h)
{
	viewport_width=w; 
	viewport_height=h; 

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0,0,viewport_width,viewport_height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0,viewport_width, 0,viewport_height);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
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
	glutInitWindowSize(viewport_width,viewport_height);
	glutInitWindowPosition(0,0);
	glutCreateWindow(":-)");

	glutDisplayFunc(my_display);
	glutReshapeFunc(my_resize);
	glutIdleFunc(my_idle);
	glutSetCursor(GLUT_CURSOR_CROSSHAIR);

	// initialize necessary OpenGL extensions
	glewInit();

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glShadeModel(GL_SMOOTH);
	glViewport(0,0,viewport_width,viewport_height);

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

void exercise7(int argc, char *argv[])
{
	//initializeCUDA(deviceProp);

	init_gl(argc, argv);

	load_texture("lena.png");

	prepare_pbo();

	init_cud_atex();

	//start rendering mainloop
    glutMainLoop();
    atexit(release_resources);
}
