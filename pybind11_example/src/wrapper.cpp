
#include "wrapper.h"
#include "transpose.cuh"
#include <iostream>


py::array_t<float> transpose_cpp(py::array_t<float>& matrix) {
    auto   ibuf      = matrix.request();
    float* input_ptr = static_cast<float*>(ibuf.ptr);
    int    n         = matrix.shape()[0];
    int    m         = matrix.shape()[1];

    auto   result     = py::array_t<float>(ibuf.size);
    auto   obuf       = result.request();
    float* output_ptr = static_cast<float*>(obuf.ptr);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            output_ptr[i * n + j] = input_ptr[j * n + i];

    result.resize({m, n});
    return result;
}


py::array_t<float> transpose_cuda(py::array_t<float>& matrix) {
    auto   ibuf      = matrix.request();
    float* input_ptr = static_cast<float*>(ibuf.ptr);
    float  m         = matrix.shape()[0];
    float  n         = matrix.shape()[1];

    auto   result     = py::array_t<float>(ibuf.size);
    auto   obuf       = result.request();
    float* output_ptr = static_cast<float*>(obuf.ptr);

    cuda_accelerations::transpose(input_ptr, output_ptr, n, 32, 32);

    result.resize({m, n});
    return result;
}