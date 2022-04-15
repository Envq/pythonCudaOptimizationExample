#ifndef TRANSPOSE_H
#define TRANSPOSE_H

namespace cuda_accelerations {
void transpose(float* h_input, float* h_output, int size, int block_size_x,
               int block_size_y);
}


#endif