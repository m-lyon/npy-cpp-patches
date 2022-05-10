// Copyright (c) 2022 Matthew Lyon. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "src/patcher.hpp"


PYBIND11_MODULE(npy_patcher, m) {
    pybind11::class_<Patcher<double>>(m, "PatcherDouble")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<double>::get_data_shape, "Get the data shape")
        .def("debug_vars", &Patcher<double>::debug_vars, "Initialise vars for debug")
        .def("get_patch", &Patcher<double>::get_patch, "Read a patch from file")
        .def("get_data_strides", &Patcher<double>::get_data_strides, "Get the data strides")
        .def("get_patch_numbers", &Patcher<double>::get_patch_numbers, "Get the patch numbers")
        .def("get_patch_strides", &Patcher<double>::get_patch_strides, "Get the patch strides")
        .def("get_shift_lengths", &Patcher<double>::get_shift_lengths, "Get the shift lengths")
        .def("get_stream_start", &Patcher<double>::get_stream_start,
             "Get the patch starting position in stream")
        .def("get_padding", &Patcher<double>::get_padding, "Get padding list");

    pybind11::class_<Patcher<float>>(m, "PatcherFloat")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<float>::get_data_shape, "Get the data shape")
        .def("debug_vars", &Patcher<float>::debug_vars, "Initialise vars for debug")
        .def("get_patch", &Patcher<float>::get_patch, "Read a patch from file")
        .def("get_data_strides", &Patcher<float>::get_data_strides, "Get the data strides")
        .def("get_patch_numbers", &Patcher<float>::get_patch_numbers, "Get the patch numbers")
        .def("get_patch_strides", &Patcher<float>::get_patch_strides, "Get the patch strides")
        .def("get_shift_lengths", &Patcher<float>::get_shift_lengths, "Get the shift lengths")
        .def("get_stream_start", &Patcher<float>::get_stream_start,
             "Get the patch starting position in stream")
        .def("get_padding", &Patcher<float>::get_padding, "Get padding list");

    pybind11::class_<Patcher<int>>(m, "PatcherInt")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<int>::get_data_shape, "Get the data shape")
        .def("debug_vars", &Patcher<int>::debug_vars, "Initialise vars for debug")
        .def("get_patch", &Patcher<int>::get_patch, "Read a patch from file")
        .def("get_data_strides", &Patcher<int>::get_data_strides, "Get the data strides")
        .def("get_patch_numbers", &Patcher<int>::get_patch_numbers, "Get the patch numbers")
        .def("get_patch_strides", &Patcher<int>::get_patch_strides, "Get the patch strides")
        .def("get_shift_lengths", &Patcher<int>::get_shift_lengths, "Get the shift lengths")
        .def("get_stream_start", &Patcher<int>::get_stream_start,
             "Get the patch starting position in stream")
        .def("get_padding", &Patcher<int>::get_padding, "Get padding list");

    pybind11::class_<Patcher<int64_t>>(m, "PatcherLong")
        .def(pybind11::init<>())
        .def("get_data_shape", &Patcher<int64_t>::get_data_shape, "Get the data shape")
        .def("debug_vars", &Patcher<int64_t>::debug_vars, "Initialise vars for debug")
        .def("get_patch", &Patcher<int64_t>::get_patch, "Read a patch from file")
        .def("get_data_strides", &Patcher<int64_t>::get_data_strides, "Get the data strides")
        .def("get_patch_numbers", &Patcher<int64_t>::get_patch_numbers, "Get the patch numbers")
        .def("get_patch_strides", &Patcher<int64_t>::get_patch_strides, "Get the patch strides")
        .def("get_shift_lengths", &Patcher<int64_t>::get_shift_lengths, "Get the shift lengths")
        .def("get_stream_start", &Patcher<int64_t>::get_stream_start,
             "Get the patch starting position in stream")
        .def("get_padding", &Patcher<int64_t>::get_padding, "Get padding list");
}
