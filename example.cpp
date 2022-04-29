#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "patcher.hpp"


PYBIND11_MODULE(example, m) {
    pybind11::class_<Patcher<double>>(m, "PatcherDouble")
        .def("get_patch", &Patcher<double>::get_patch, "get patch");
}