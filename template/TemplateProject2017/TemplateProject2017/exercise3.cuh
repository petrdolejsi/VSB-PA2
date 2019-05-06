#pragma once

void exercise3();
__global__ void add(const float3* __restrict__ dFinalForce, const unsigned int noRainDrops, float3* __restrict__ dRainDrops);
__global__ void reduce(const float3 * __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce);
float3 *createData(const unsigned int length);
void printData(const float3 *data, const unsigned int length);