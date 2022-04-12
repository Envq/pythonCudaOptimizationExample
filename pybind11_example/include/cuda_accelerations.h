#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;


py::array_t<double> transpose_cpp(py::array_t<double>& matrix);

void transpose_cpp2(py::array_t<double>& input, py::array_t<double>& output);


PYBIND11_MODULE(cuda_accelerations, mod) {
    mod.def("transpose_cpp", &transpose_cpp);
    mod.def("transpose_cpp2", &transpose_cpp2);
}
