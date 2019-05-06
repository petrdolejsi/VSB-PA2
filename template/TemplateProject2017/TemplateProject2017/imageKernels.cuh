#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Converts color a float [pitch] linear memory - PREPARED FOR 2D CUDA BLOCK in generall,
/// 			but blockDim.y can be 1 as well.</summary>
/// <typeparam name="unsigned int srcBPP"> Source bits per pixel. </param>
/// <param name="src">   	Source data. </param>
/// <param name="srcWidth"> The width. </param>
/// <param name="srcHeight">The height. </param>
/// <param name="srcHeight">The pitch of src. </param>
/// <param name="dstPitch">	The pitch of dst. </param>
/// <param name="dst">   	[in,out] If non-null, destination for the. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned int srcBPP>__global__ void colorToFloat(const unsigned char * __restrict__ src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* __restrict__ dst )
{
	const auto tx = blockIdx.x * blockDim.x + threadIdx.x;
	const auto ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((tx < srcWidth) && (ty < srcHeight))
	{
		unsigned int dst_offset = ty * dstPitch + tx;
		unsigned int src_offset = (ty * srcPitch + tx) * srcBPP/8;
		unsigned int value = 0;

		if (srcBPP>=8)
		{
			value = src[src_offset++];
		}
		if (srcBPP>=16)
		{
			value = (value<<8) | src[src_offset++];
		}
		if (srcBPP>=24)
		{
			value = (value<<8) | src[src_offset++];
		}
		if (srcBPP>=32)
		{
			value = (value<<8) | src[src_offset++];
		}
		dst[dst_offset] = (float)value;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned int srcBPP>__global__ void colorToUchar4(const uchar4 * __restrict__ src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, uchar4* __restrict__ dst)
{
	const auto tx = blockIdx.x * blockDim.x + threadIdx.x;
	const auto ty = blockIdx.y * blockDim.y + threadIdx.y;

	if ((tx < srcWidth) && (ty < srcHeight))
	{
		const auto dst_offset = ty * dstPitch + tx;
		auto src_offset = (ty * srcPitch + tx) * srcBPP / 8;
		uchar4 value;
		value.w = 0;
		value.x = 0;
		value.y = 0;
		value.z = 0;

		if (srcBPP >= 8)
		{
			value.w = src[src_offset++].w;
			value.x = src[src_offset].x;
			value.y = src[src_offset].y;
			value.z = src[src_offset].z;
		}
		if (srcBPP >= 16)
		{
			value.w = (value.w << 8) | src[src_offset++].w;
			value.x = (value.x << 8) | src[src_offset].x;
			value.y = (value.y << 8) | src[src_offset].y;
			value.z = (value.z << 8) | src[src_offset].z;
		}
		if (srcBPP >= 24)
		{
			value.w = (value.w << 8) | src[src_offset++].w;
			value.x = (value.x << 8) | src[src_offset].x;
			value.y = (value.y << 8) | src[src_offset].y;
			value.z = (value.z << 8) | src[src_offset].z;
		}
		if (srcBPP >= 32)
		{
			value.w = (value.w << 8) | src[src_offset++].w;
			value.x = (value.x << 8) | src[src_offset].x;
			value.y = (value.y << 8) | src[src_offset].y;
			value.z = (value.z << 8) | src[src_offset].z;
		}
		dst[dst_offset] = value;
	}
}