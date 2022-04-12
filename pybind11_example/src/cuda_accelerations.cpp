#include "cuda_accelerations.h"
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


void transpose_cpp2(py::array_t<double>& input, py::array_t<double>& output) {
    auto    ibuf       = input.request();
    auto    obuf       = output.request();
    double* input_ptr  = static_cast<double*>(ibuf.ptr);
    double* output_ptr = static_cast<double*>(obuf.ptr);

    _transpose_cpp(input_ptr, output_ptr, input.shape()[0]);
}