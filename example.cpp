#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "patcher.hpp"


PYBIND11_MODULE(example, m) {
    pybind11::class_<Patcher<double>>(m, "PatcherDouble")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<double>::get_data_shape, "get the data shape")
        .def("get_patch", &Patcher<double>::get_patch, "read the patch");

    pybind11::class_<Patcher<float>>(m, "PatcherFloat")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<float>::get_data_shape, "get the data shape")
        .def("get_patch", &Patcher<float>::get_patch, "read the patch");
}

/*
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
*/