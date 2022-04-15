#include <stdio.h>


extern "C" __global__ void nop(const float* d_input, float* d_output,
                               const int m, const int n) {
}


extern "C" __global__ void transpose(const float* d_input, float* d_output,
                                     const int m, const int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < m) {
        d_output[col * m + row] = d_input[row * n + col];
    }
}

extern "C" __global__ void transpose_shm(const float* d_input, float* d_output,
                                         const int m, const int n) {
    const int        TILE = 32;
    __shared__ float buffer[TILE][TILE];

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

extern "C" __global__ void transpose_shm3(const float* d_input, float* d_output,
                                          const int m, const int n) {
    const int        TILE       = 32;
    const int        BLOCK_ROWS = TILE / blockDim.y;
    __shared__ float buffer[TILE][TILE + 1];

    // read matrix in linear order
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    // if ((col < n) && (row < m))
    for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
        buffer[threadIdx.y + offset][threadIdx.x] =
            d_input[(row + offset) * n + col];
    }
    __syncthreads();

    // write transposed matrix in linear order
    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    // if ((col < m) && (row < n))
    for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
        // transpose is done with buffer
        d_output[(row + offset) * m + col] =
            buffer[threadIdx.x][threadIdx.y + offset];
    }
}

extern "C" __global__ void transpose_shm2(const float* d_input, float* d_output,
                                          const int m, const int n) {
    const int        BLOCK_ROWS = 8;
    const int        TILE       = 32;
    __shared__ float tile[TILE][TILE + 1];

    int x     = blockIdx.x * TILE + threadIdx.x;
    int y     = blockIdx.y * TILE + threadIdx.y;
    int width = gridDim.x * TILE;

    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = d_input[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE + threadIdx.y;

    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        d_output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}