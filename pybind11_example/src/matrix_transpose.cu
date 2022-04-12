#include "CheckError.cuh"
#include "matrix_transpose.cuh"


__global__ void matrix_transpose_kernel(const double* d_matrix_in,
                                        double* d_matrix_out, int N) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    d_matrix_out[row * N + col] = d_matrix_in[col * N + row];
}


void matrix_transpose(double* h_input, double* h_output, int size,
                      int block_size_x, int block_size_y) {
    // DEVICE MEMORY ALLOCATION
    double *     d_input, *d_output;
    const size_t size_byte = size * size * sizeof(double);
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
    matrix_transpose_kernel<<<DimGrid, DimBlock>>>(d_input, d_output, size);
    CHECK_CUDA_ERROR

    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL(cudaMemcpy(h_output, d_output, size_byte, cudaMemcpyDeviceToHost))

    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL(cudaFree(d_input))
    SAFE_CALL(cudaFree(d_output))

    cudaDeviceReset();
}
