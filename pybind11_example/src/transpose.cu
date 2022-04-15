#include "CheckError.cuh"
#include "transpose.cuh"
#include <iostream>


__global__ void matrix_transpose_kernel(const float* d_matrix_in,
                                        float* d_matrix_out, int N) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    d_matrix_out[row * N + col] = d_matrix_in[col * N + row];
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

    std::cout << "DimGrid: " << DimGrid.x << DimGrid.y << DimGrid.z
              << std::endl;
    std::cout << "DimBlock: " << DimBlock.x << DimBlock.y << DimBlock.z
              << std::endl;

    // DEVICE EXECUTION
    matrix_transpose_kernel<<<DimGrid, DimBlock>>>(d_input, d_output, size);
    CHECK_CUDA_ERROR

    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL(cudaMemcpy(h_output, d_output, size_byte, cudaMemcpyDeviceToHost))

    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL(cudaFree(d_input))
    SAFE_CALL(cudaFree(d_output))

    cudaDeviceReset();
}
