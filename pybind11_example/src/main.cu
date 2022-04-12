#include "matrix_transpose.cuh"
#include <iostream>


void printMatrix(double* matrix, int size) {
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
    double* h_input  = new double[size * size]{};
    double* h_output = new double[size * size]{};

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
    matrix_transpose(h_input, h_output, size, block_size_x, block_size_y);
    printMatrix(h_output, size);

    // HOST MEMORY DEALLOCATION
    delete[] h_input;
    delete[] h_output;
}