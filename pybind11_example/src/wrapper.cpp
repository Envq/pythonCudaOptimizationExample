
#include "wrapper.h"
#include "matrix_transpose.cuh"
#include <iostream>


void _transpose_cpp(double* input, double* output, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            output[i * size + j] = input[j * size + i];
        }
    }
}


py::array_t<double> transpose_cpp(py::array_t<double>& matrix) {
    auto    ibuf      = matrix.request();
    double* input_ptr = static_cast<double*>(ibuf.ptr);
    int     m         = matrix.shape()[0];
    int     n         = matrix.shape()[1];

    auto    result     = py::array_t<double>(ibuf.size);
    auto    obuf       = result.request();
    double* output_ptr = static_cast<double*>(obuf.ptr);

    _transpose_cpp(input_ptr, output_ptr, n);

    result.resize({m, n});
    return result;
}


py::array_t<double> transpose_cuda(py::array_t<double>& matrix) {
    auto    ibuf      = matrix.request();
    double* input_ptr = static_cast<double*>(ibuf.ptr);
    double  m         = matrix.shape()[0];
    double  n         = matrix.shape()[1];

    auto    result     = py::array_t<double>(ibuf.size);
    auto    obuf       = result.request();
    double* output_ptr = static_cast<double*>(obuf.ptr);

    matrix_transpose(input_ptr, output_ptr, n, 4, 4);
    // _transpose_cpp(input_ptr, output_ptr, n);

    result.resize({m, n});
    return result;
}