#ifndef CUDA_ACCELERATIONS_H
#define CUDA_ACCELERATIONS_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;


py::array_t<double> transpose_cpp(py::array_t<double>& matrix);
py::array_t<double> transpose_cuda(py::array_t<double>& matrix);


PYBIND11_MODULE(cuda_accelerations, mod) {
    mod.def("transpose_cpp", &transpose_cpp);
    mod.def("transpose_cuda", &transpose_cuda);
}

#endif