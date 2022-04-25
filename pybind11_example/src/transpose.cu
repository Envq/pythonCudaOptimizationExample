#include "CheckError.cuh"
#include "transpose.cuh"


const int TILE = 32;

extern "C" __global__ void matrix_transpose_kernel(const float* d_input,
                                                   float* d_output, const int m,
                                                   const int n) {
    __shared__ float buffer[TILE][TILE + 1];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if ((col < n) && (row < m)) {
        buffer[threadIdx.y][threadIdx.x] = d_input[row * n + col];
    }
    __syncthreads();

    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    if ((col < m) && (row < n)) {
        d_output[row * m + col] = buffer[threadIdx.x][threadIdx.y];
    }
}


void cuda_accelerations::transpose(float* h_input, float* h_output, int size,
                                   int block_size_x, int block_size_y) {
    // DEVICE MEMORY ALLOCATION
    float *      d_input, *d_output;
    const size_t size_byte = size * size * sizeof(float);
    SAFE_CALL(cudaMalloc(&d_input, size_byte))
    SAFE_CALL(cudaMalloc(&d_output, size_byte))

    // COPY DATA FROM HOST TO DEVICE
    SAFE_CALL(cudaMemcpy(d_input, h_input, size_byte, cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_output, h_output, size_byte, cudaMemcpyHostToDevice))

    // DEVICE INIT
    dim3 DimGrid(size / block_size_x, size / block_size_y, 1);
    if (size % block_size_x)
        DimGrid.x++;
    if (size % block_size_y)
        DimGrid.y++;
    dim3 DimBlock(block_size_x, block_size_y, 1);

    // DEVICE EXECUTION
    matrix_transpose_kernel<<<DimGrid, DimBlock>>>(d_input, d_output, size,
                                                   size);
    CHECK_CUDA_ERROR

    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL(cudaMemcpy(h_output, d_output, size_byte, cudaMemcpyDeviceToHost))

    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL(cudaFree(d_input))
    SAFE_CALL(cudaFree(d_output))

    cudaDeviceReset();
}
