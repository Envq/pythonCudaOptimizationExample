#include "transpose.cuh"
#include <iostream>


void printMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    int size         = 8;
    int tile         = 4;
    int block_size_x = tile;
    int block_size_y = tile;

    // PRINT INFO
    std::cout << "SIZE:          " << size << std::endl;
    std::cout << "BLOCK_SIZE_X:  " << block_size_x << std::endl;
    std::cout << "BLOCK_SIZE_Y:  " << block_size_y << "\n" << std::endl;

    // HOST MEMORY ALLOCATION
    float* h_input  = new float[size * size]{};
    float* h_output = new float[size * size]{};

    // HOST INITILIZATION
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (j <= i) {
                h_input[i * size + j] = 3.0f;
            }
        }
    }

    // CUDA EXECUTION
    printMatrix(h_input, size);
    cuda_accelerations::transpose(h_input, h_output, size, block_size_x,
                                  block_size_y);
    printMatrix(h_output, size);

    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
}