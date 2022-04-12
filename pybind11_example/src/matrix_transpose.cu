#include "CheckError.cuh"
#include <iostream>



__global__ void matrixTransposeKernel(const int *d_matrix_in, int *d_matrix_out,
                                      int N) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    d_matrix_out[row * N + col] = d_matrix_in[col * N + row];
}


void cudaTranspose(int *h_input, int *h_output, int size, int block_size_x, int block_size_y) {
    // DEVICE MEMORY ALLOCATION
    int *d_input, *d_output;
    const size_t size_byte = size * size * sizeof(int);
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
    matrixTransposeKernel<<<DimGrid, DimBlock>>>(d_input, d_output, size);
    CHECK_CUDA_ERROR

    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL(cudaMemcpy(h_output, d_output, size_byte, cudaMemcpyDeviceToHost))

    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL(cudaFree(d_input))
    SAFE_CALL(cudaFree(d_output))

    cudaDeviceReset();
}


void printMatrix(int *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


int main(int argc, char *argv[]) {
    int size = 8;
    int tile = 4;
    int block_size_x = tile;
    int block_size_y = tile;

    // PRINT INFO
    std::cout << "SIZE:          " << size << std::endl;
    std::cout << "BLOCK_SIZE_X:  " << block_size_x << std::endl;
    std::cout << "BLOCK_SIZE_Y:  " << block_size_y << "\n" << std::endl;

    // HOST MEMORY ALLOCATION
    int *h_input = new int[size * size]{};
    int *h_output = new int[size * size]{};

    // HOST INITILIZATION
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (j <= i) {
                h_input[i * size + j] = 3;
            }
        }
    }

    // CUDA EXECUTION
    printMatrix(h_input, size);
    cudaTranspose(h_input, h_output, size, block_size_x, block_size_y);
    printMatrix(h_output, size);

    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
}