#include <stdio.h>


const int TILE = 32;


extern "C" __global__ void transpose(const float* d_input, float* d_output,
                                     const int m, const int n) {
    __shared__ float buffer[TILE][TILE + 1];

    // read matrix in linear order
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if ((col < n) && (row < m)) {
        buffer[threadIdx.y][threadIdx.x] = d_input[row * n + col];
    }
    __syncthreads();

    // write transposed matrix in linear order
    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    if ((col < m) && (row < n)) {
        // transpose is done with buffer
        d_output[row * m + col] = buffer[threadIdx.x][threadIdx.y];
    }
}
