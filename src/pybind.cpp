// Copyright (c) 2022 Matthew Lyon. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "src/patcher.hpp"


PYBIND11_MODULE(npy_patcher, m) {
    pybind11::class_<Patcher<double>>(m, "PatcherDouble")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<double>::get_data_shape, "get the data shape")
        .def("debug_vars", &Patcher<double>::debug_vars, "init vars for debug")
        .def("get_patch", &Patcher<double>::get_patch, "read the patch")
        .def("get_data_strides", &Patcher<double>::get_data_strides, "get the data strides")
        .def("get_patch_strides", &Patcher<double>::get_patch_strides, "get the patch strides")
        .def("get_shift_lengths", &Patcher<double>::get_shift_lengths, "get the shift lengths")
        .def("get_stream_start", &Patcher<double>::get_stream_start, "get the patch starting position in stream")
        .def("get_padding", &Patcher<double>::get_padding, "gets padding list");

    pybind11::class_<Patcher<float>>(m, "PatcherFloat")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<float>::get_data_shape, "get the data shape")
        .def("debug_vars", &Patcher<float>::debug_vars, "init vars for debug")
        .def("get_patch", &Patcher<float>::get_patch, "read the patch")
        .def("get_data_strides", &Patcher<float>::get_data_strides, "get the data strides")
        .def("get_patch_strides", &Patcher<float>::get_patch_strides, "get the patch strides")
        .def("get_shift_lengths", &Patcher<float>::get_shift_lengths, "get the shift lengths")
        .def("get_stream_start", &Patcher<float>::get_stream_start, "get the patch starting position in stream")
        .def("get_padding", &Patcher<float>::get_padding, "gets padding list");

    pybind11::class_<Patcher<int>>(m, "PatcherInt")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<int>::get_data_shape, "get the data shape")
        .def("debug_vars", &Patcher<int>::debug_vars, "init vars for debug")
        .def("get_patch", &Patcher<int>::get_patch, "read the patch")
        .def("get_data_strides", &Patcher<int>::get_data_strides, "get the data strides")
        .def("get_patch_strides", &Patcher<int>::get_patch_strides, "get the patch strides")
        .def("get_shift_lengths", &Patcher<int>::get_shift_lengths, "get the shift lengths")
        .def("get_stream_start", &Patcher<int>::get_stream_start, "get the patch starting position in stream")
        .def("get_padding", &Patcher<int>::get_padding, "gets padding list");

    pybind11::class_<Patcher<long>>(m, "PatcherLong")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<long>::get_data_shape, "get the data shape")
        .def("debug_vars", &Patcher<long>::debug_vars, "init vars for debug")
        .def("get_patch", &Patcher<long>::get_patch, "read the patch")
        .def("get_data_strides", &Patcher<long>::get_data_strides, "get the data strides")
        .def("get_patch_strides", &Patcher<long>::get_patch_strides, "get the patch strides")
        .def("get_shift_lengths", &Patcher<long>::get_shift_lengths, "get the shift lengths")
        .def("get_stream_start", &Patcher<long>::get_stream_start, "get the patch starting position in stream")
        .def("get_padding", &Patcher<long>::get_padding, "gets padding list");
}
