#pragma once

void exercise2();
__global__ void fill(int* matrix, size_t rows, size_t cols, size_t pitch);
__global__ void increment(int* matrix, size_t rows, size_t cols, size_t pitch);