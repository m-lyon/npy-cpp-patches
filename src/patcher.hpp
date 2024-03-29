// Copyright (c) 2022 Matthew Lyon. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef PATCHER_HPP_
#define PATCHER_HPP_

#include <fstream>  // std::ifstream
#include <sstream>  // std::ostringstream
#include <string>   // std::string
#include <vector>   // std::vector

#include "src/npy_header.hpp"

// TODO(m-lyon): Move to stateful reading of object, i.e. initialise with filepath & patch_shape.
//                 then call get_patch with qspace_index & patch_num.
//               Will need to ensure deconstructor closes the file.
//               Should make filepath & patch_shape constants to ensure object is not
//                 reused for different file.

// TODO(m-lyon): Remove after debug
template <typename T>
void print_vector(const std::vector<T> &data) {
    if (data.empty()) {
        std::cout << "empty vector given" << std::endl;
        return;
    }
    std::cout << "(";
    for (size_t i = 0; i < data.size() - 1; i++) {
        std::cout << data[i] << ", ";
    }
    std::cout << data[data.size() - 1] << ")" << std::endl;
}

/**
 * @brief Patcher object
 *
 * @tparam T datatype of data found within fpath
 */
template <typename T>
class Patcher {
  private:
    std::string filepath;
    std::ifstream stream;
    std::vector<T> patch;
    std::vector<size_t> data_shape, qspace_index, patch_shape, patch_stride, patch_num;
    std::vector<size_t> num_patches, padding, data_strides, patch_byte_strides, shifts;
    std::vector<size_t> extra_padding;
    std::vector<size_t> patch_num_offset;
    size_t patch_size, start, pos;
    bool has_run = false;
    char *buf;
    void set_init_vars(const std::string &, const std::vector<size_t> &,
                       const std::vector<size_t> &, const std::vector<size_t> &,
                       const std::vector<size_t> &, const std::vector<size_t> &);
    void set_runtime_vars(size_t);
    void set_patch_numbers(size_t);
    void set_patch_size();
    void open_file();
    void set_padding();
    void set_strides();
    void set_shift_lengths();
    void set_num_of_patches();
    void move_stream_to_start();
    void read_patch();
    void read_nd_slice(const unsigned int);
    void read_slice();
    void set_extra_padding();
    void set_patch_num_offset();
    void sanity_check();

  public:
    Patcher();
    std::vector<T> get_patch(const std::string &, const std::vector<size_t> &, std::vector<size_t>,
                             std::vector<size_t>, size_t, std::vector<size_t>, std::vector<size_t>);
    void debug_vars(const std::string &, const std::vector<size_t> &, std::vector<size_t>,
                    std::vector<size_t>, size_t, std::vector<size_t>, std::vector<size_t>);
    size_t get_patch_size();
    size_t get_stream_start();
    std::vector<size_t> get_data_shape();
    std::vector<size_t> get_padding();
    std::vector<size_t> get_data_strides();
    std::vector<size_t> get_patch_strides();
    std::vector<size_t> get_num_patches();
    std::vector<size_t> get_shift_lengths();
    std::vector<size_t> get_patch_numbers();
};

template <typename T>
Patcher<T>::Patcher() {}

/**
 * @brief Sets internal variables, these are set before any file reading is necessary.
 *
 * @tparam T datatype of data found within fpath
 * @param fpath filepath for .npy data file
 * @param qidx qspace index (0th index in file)
 * @param pshape patch shape
 * @param pnum patch number for each patch dimension
 */
template <typename T>
void Patcher<T>::set_init_vars(const std::string &fpath, const std::vector<size_t> &qidx,
                               const std::vector<size_t> &pshape,
                               const std::vector<size_t> &pstride,
                               const std::vector<size_t> &padding,
                               const std::vector<size_t> &pnum_offset) {
    filepath = fpath;
    qspace_index = qidx;
    patch_shape = pshape;
    patch_stride = pstride;
    extra_padding = padding;
    patch_num_offset = pnum_offset;
    std::reverse(patch_shape.begin(), patch_shape.end());
    std::reverse(patch_stride.begin(), patch_stride.end());
    set_patch_num_offset();

    // Init/reset patch object
    set_patch_size();
    if (has_run) {
        patch.clear();
    }
    patch.resize(patch_size, 0);
}

/**
 * @brief Opens npy file ready for data extraction. Reads and parses header.
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::open_file() {
    // Open file
    stream.open(filepath, std::ifstream::binary);

    // Read and parse header
    std::string header_s = npy_header::read_header(stream);
    start = stream.tellg();
    npy_header::header_t header = npy_header::parse_header(header_s);
    data_shape = header.shape;
    std::reverse(data_shape.begin(), data_shape.end());

    // Data validation
    if (!stream) {
        throw std::runtime_error("IO Error: failed to open " + filepath);
    }

    static_assert(npy_header::has_typestring<T>::value, "Unrecognised datatype in file.");
    if (header.dtype.tie() != npy_header::has_typestring<T>::dtype.tie()) {
        throw std::runtime_error("Type mismatch between class and file.");
    }

    if (header.fortran_order) {
        throw std::runtime_error("Fortran data order extraction not currently implemented.");
    }
}

/**
 * @brief Closes file after finished extracting patch.
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::sanity_check() {
    if (!stream) {
        throw std::runtime_error("Failed to get patch within " + filepath);
    }
    stream.close();
}

/**
 * @brief Reverses extra padding vector, sets to 0 if empty vector given.
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::set_extra_padding() {
    if (extra_padding.empty()) {
        extra_padding = std::vector<size_t>(padding.size(), 0);
        return;
    }
    if (extra_padding.size() != padding.size()) {
        std::runtime_error("Extra padding given is invalid shape.");
    }
    std::vector<size_t> reversed_padding;
    reversed_padding.reserve(extra_padding.size());

    size_t r;
    for (size_t i = 0; i < extra_padding.size(); i++) {
        if (i % 2 == 0) {
            r = extra_padding.size() - 2 - i;
        } else {
            r = extra_padding.size() - i;
        }
        reversed_padding.push_back(extra_padding[r]);
    }
    extra_padding = reversed_padding;
}

template <typename T>
void Patcher<T>::set_patch_num_offset() {
    if (patch_num_offset.empty()) {
        patch_num_offset = std::vector<size_t>(patch_shape.size(), 0);
        return;
    }
    if (patch_num_offset.size() != patch_shape.size()) {
        std::runtime_error("Patch num offset given is invalid shape.");
    }
    std::reverse(patch_num_offset.begin(), patch_num_offset.end());
}

/**
 * @brief Calculates padding needed to split data into patches given by
 *      patch_shape and patch_stride
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::set_padding() {
    padding.resize(patch_shape.size() * 2, 0);

    set_extra_padding();

    unsigned int total_pad, total_size;
    float num;

    // Iterate over dimensions
    for (size_t i = 0; i < patch_shape.size(); i++) {
        // Get total padding to ensure valid shape
        if (data_shape[i] <= patch_shape[i]) {
            total_pad = patch_shape[i] - data_shape[i];
        } else {
            num = std::ceil(static_cast<float>(data_shape[i] - patch_shape[i]) /
                            static_cast<float>(patch_stride[i]));
            total_pad = ((static_cast<unsigned int>(num) * patch_stride[i]) + patch_shape[i]) -
                        data_shape[i];
        }

        // Combine with extra padding and calculate padding either side of dimension
        if (total_pad == 0) {
            padding[i * 2] = extra_padding[i * 2];
            padding[(i * 2) + 1] = extra_padding[(i * 2) + 1];
        } else {
            if (total_pad % 2 == 0) {
                padding[i * 2] = (total_pad / 2) + extra_padding[i * 2];
            } else {
                padding[i * 2] = (total_pad / 2) + 1 + extra_padding[i * 2];
            }
            padding[(i * 2) + 1] = (total_pad / 2) + extra_padding[(i * 2) + 1];
        }

        // Validate the padding, shape, and stride are still valid.
        std::ostringstream oss;
        if ((padding[i * 2] > patch_shape[i]) || (padding[(i * 2) + 1] > patch_shape[i])) {
            oss << "Resultant padding in dim " << i << ", ";
            if ((padding[i * 2] > patch_shape[i])) {
                oss << "left side (" << padding[i * 2] << ") ";
            } else {
                oss << "right side (" << padding[(i * 2) + 1] << ") ";
            }
            oss << "is greater than patch shape (" << patch_shape[i] << ").";
            throw std::runtime_error(oss.str());
        }
        total_size = data_shape[i] + padding[i * 2] + padding[(i * 2) + 1];
        if (total_size < patch_shape[i]) {
            oss << "Total padded size in dim " << i << " (" << total_size << ") ";
            oss << "is less than patch size (" << patch_shape[i] << ").";
            throw std::runtime_error(oss.str());
        }
        if ((total_size > patch_shape[i]) &&
            (((total_size - patch_shape[i]) % patch_stride[i]) != 0)) {
            oss << "Padding is invalid. total padded size in dim " << i << " (" << total_size;
            oss << ") ";
            oss << "while patch shape is " << patch_shape[i] << " and patch stride is ";
            oss << patch_stride[i];
            throw std::runtime_error(oss.str());
        }
    }
}

/**
 * @brief Get the calculated padding.
 *
 * @tparam T datatype of data found within filepath
 * @return std::vector<size_t> Padding
 */
template <typename T>
std::vector<size_t> Patcher<T>::get_padding() {
    std::vector<size_t> out(padding.size());
    for (size_t i = 0; i < out.size() / 2; i++) {
        out[i * 2] = padding[out.size() - 2 - (i * 2)];
        out[(i * 2) + 1] = padding[out.size() - 1 - (i * 2)];
    }
    return out;
}

/**
 * @brief Sets total patch size
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::set_patch_size() {
    patch_size = 1;
    for (size_t i : patch_shape) {
        patch_size *= i;
    }
    patch_size *= qspace_index.size();
}

/**
 * @brief Gets total patch size
 *
 * @tparam T datatype of data found within filepath
 * @return size_t Total patch size
 */
template <typename T>
size_t Patcher<T>::get_patch_size() {
    return patch_size;
}

/**
 * @brief Sets data_strides vector
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::set_strides() {
    data_strides.resize(patch_shape.size() + 1, 0);
    data_strides[0] = sizeof(T);  // 0th dimension moves linearly
    for (size_t i = 1; i <= patch_shape.size(); i++) {
        data_strides[i] = data_shape[i - 1] * data_strides[i - 1];
    }

    patch_byte_strides.resize(patch_shape.size(), 0);
    patch_byte_strides[0] = data_strides[0];
    for (size_t i = 1; i < patch_shape.size(); i++) {
        patch_byte_strides[i] = patch_shape[i - 1] * patch_byte_strides[i - 1];
    }
}

template <typename T>
std::vector<size_t> Patcher<T>::get_data_strides() {
    std::vector<size_t> out(data_strides.size());
    std::reverse_copy(data_strides.begin(), data_strides.end(), out.begin());
    return out;
}

template <typename T>
std::vector<size_t> Patcher<T>::get_patch_strides() {
    std::vector<size_t> out(patch_byte_strides.size());
    std::reverse_copy(patch_byte_strides.begin(), patch_byte_strides.end(), out.begin());
    return out;
}

/**
 * @brief Returns the maximum number of patches in each dimension
 *
 * @tparam T datatype of data found within filepath
 * @return std::vector<size_t> Number of patches
 */
template <typename T>
std::vector<size_t> Patcher<T>::get_num_patches() {
    std::vector<size_t> out(num_patches.size());
    std::reverse_copy(num_patches.begin(), num_patches.end(), out.begin());
    return out;
}

/**
 * @brief Set the num_patches object
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::set_num_of_patches() {
    num_patches.resize(data_shape.size() - 1);
    for (size_t i = 0; i < num_patches.size(); i++) {
        if (data_shape[i] <= patch_shape[i]) {
            num_patches[i] = 1;
        } else {
            num_patches[i] =
                ((data_shape[i] + padding[2 * i] + padding[(2 * i) + 1] - patch_shape[i]) /
                 patch_stride[i]) +
                1;
        }
    }
}

/**
 * @brief Converts patch number into patch number across each dimension
 *
 * @tparam T datatype of data found within filepath
 * @param pnum Patch number to be converted to patch num in each dimension
 */
template <typename T>
void Patcher<T>::set_patch_numbers(size_t pnum) {
    size_t max_patch_num = 1;
    for (size_t i = 0; i < num_patches.size(); i++) {
        max_patch_num *= num_patches[i];
    }

    // Patch number validation
    if (pnum >= max_patch_num) {
        std::ostringstream oss;
        oss << "Max patch index: " << max_patch_num - 1 << ", " << pnum << " given.";
        throw std::runtime_error(oss.str());
    }

    // Reset state
    if (has_run) {
        patch_num.clear();
    }
    patch_num.resize(num_patches.size(), 0);

    // Get patch number strides
    std::vector<size_t> patch_num_strides(num_patches.size(), 1);
    for (size_t i = 1; i < num_patches.size(); i++) {
        patch_num_strides[i] = patch_num_strides[i - 1] * num_patches[i - 1];
    }

    // Increase pnum based on pnum_offset
    for (size_t i = 0; i < patch_num_offset.size() - 1; i++) {
        if (patch_num_offset[i] >= num_patches[i]) {
            std::ostringstream oss;
            oss << "Offset greater or equal to number of patches in dim " << i << ": "
                << patch_num_offset[i] << " >= " << num_patches[i];
            throw std::runtime_error(oss.str());
        }
        pnum += patch_num_strides[i] * patch_num_offset[i];
    }

    // Calculate patch number in each dimension
    for (size_t i = num_patches.size() - 1; i >= 0; i--) {
        patch_num[i] = (pnum / patch_num_strides[i]);
        pnum -= patch_num[i] * patch_num_strides[i];
        if (pnum == 0) {
            break;
        }
    }
}

/**
 * @brief Gets the patch index in each dimension
 *
 * @tparam T datatype of data found within filepath
 * @return std::vector<size_t> Patch number for each dimension
 */
template <typename T>
std::vector<size_t> Patcher<T>::get_patch_numbers() {
    std::vector<size_t> out(patch_num.size());
    std::reverse_copy(patch_num.begin(), patch_num.end(), out.begin());
    return out;
}

/**
 * @brief Moves stream pointer to start of patch
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::move_stream_to_start() {
    size_t i = 0;
    pos = 0;
    // get relative position of patched dims
    for (; i < patch_shape.size(); i++) {
        if (patch_num[i] != 0) {
            // shift minus the padding
            pos += (data_strides[i] * patch_num[i] * patch_stride[i]) -
                   (data_strides[i] * padding[2 * i]);
        }
    }
    pos += (qspace_index[0] * data_strides[i]);  // qdim
    pos += start;
    start = pos;  // update to patch start position
    stream.seekg(pos, stream.beg);
}

template <typename T>
size_t Patcher<T>::get_stream_start() {
    return start;
}

/**
 * @brief Sets actual byte shift lengths for stream/buffer
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::set_shift_lengths() {
    shifts.resize(patch_shape.size(), 0);

    for (size_t i = 0; i < shifts.size(); i++) {
        shifts[i] = data_strides[i] * patch_shape[i];
        // If start of patch
        if (patch_num[i] == 0) {
            shifts[i] -= data_strides[i] * padding[2 * i];
        }
        // If end of patch
        if (patch_num[i] == num_patches[i] - 1) {
            shifts[i] -= data_strides[i] * padding[(2 * i) + 1];
        }
    }
}

template <typename T>
std::vector<size_t> Patcher<T>::get_shift_lengths() {
    std::vector<size_t> out(shifts.size());
    std::reverse_copy(shifts.begin(), shifts.end(), out.begin());
    return out;
}

/**
 * @brief Sets variables and data after reading data header
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::set_runtime_vars(size_t pnum) {
    set_padding();
    set_strides();
    set_num_of_patches();
    set_patch_numbers(pnum);
    set_shift_lengths();
}

/**
 * @brief Reads patch into patch vector
 *
 * @tparam T datatype of data found within filepath
 */
template <typename T>
void Patcher<T>::read_patch() {
    move_stream_to_start();
    // get data pointer as char pointer
    buf = reinterpret_cast<char *>(patch.data());
    const unsigned int dim = patch_shape.size();
    for (size_t i = 0; i < qspace_index.size() - 1; i++) {
        read_nd_slice(dim - 1);
        pos -= shifts[dim - 1];
        pos += ((qspace_index[i + 1] - qspace_index[i]) * data_strides.back());
        stream.seekg(pos, stream.beg);
    }
    read_nd_slice(dim - 1);  // last slice
}

template <typename T>
void Patcher<T>::read_slice() {
    // If in first patch, and left padded region
    if ((patch_num[0] == 0) && (padding[0] > 0)) {
        buf += patch_byte_strides[0] * padding[0];
    }
    if (shifts[0] > 0) {
        // Read and shift pointers
        stream.read(buf, shifts[0]);
        buf += shifts[0];
        pos += shifts[0];
    }
    // If in last patch, and right padded region
    if ((patch_num[0] + 1 == num_patches[0]) && (padding[1] > 0)) {
        buf += patch_byte_strides[0] * padding[1];
    }
}

/**
 * @brief Reads N-dimensional slice, intended to be used recursively.
 *
 * @tparam T datatype of data found within filepath
 * @param dim Dimensionality of slice, starting at 0.
 */
template <typename T>
void Patcher<T>::read_nd_slice(const unsigned int dim) {
    if (dim == 0) {
        read_slice();
    } else {
        // Iterate over dimension
        for (size_t i = 0; i < (patch_shape[dim]); i++) {
            // If at first patch, and within left padded region
            if ((patch_num[dim] == 0) && (i < padding[2 * dim])) {
                buf += patch_byte_strides[dim];
                // If at end patch, and within right padded region
            } else if ((patch_num[dim] + 1 == num_patches[dim]) &&
                       (i >= patch_shape[dim] - padding[(2 * dim) + 1])) {
                buf += patch_byte_strides[dim];
            } else {
                read_nd_slice(dim - 1);
                pos = pos - shifts[dim - 1] + data_strides[dim];  // Shift stream position.
                stream.seekg(pos, stream.beg);
            }
        }
    }
}

/**
 * @brief Gets data shape read from header in npy file
 *
 * @tparam T datatype of data found within filepath
 * @return std::vector<size_t> Data shape
 */
template <typename T>
std::vector<size_t> Patcher<T>::get_data_shape() {
    std::vector<size_t> out(data_shape.size());
    std::reverse_copy(data_shape.begin(), data_shape.end(), out.begin());
    return out;
}

/**
 * @brief Public method to extract patch
 *
 * @tparam T datatype of data found within fpath
 * @param fpath filepath for .npy data file
 * @param qidx qspace index (0th index in file)
 * @param pshape patch shape
 * @param pstride patch stride
 * @param pnum patch number
 * @return std::vector<T> Patch data
 */
template <typename T>
std::vector<T> Patcher<T>::get_patch(const std::string &fpath, const std::vector<size_t> &qidx,
                                     std::vector<size_t> pshape, std::vector<size_t> pstride,
                                     size_t pnum, std::vector<size_t> padding,
                                     std::vector<size_t> pnum_offset) {
    set_init_vars(fpath, qidx, pshape, pstride, padding, pnum_offset);
    open_file();
    set_runtime_vars(pnum);
    read_patch();
    sanity_check();
    has_run = true;

    return patch;
}

template <typename T>
void Patcher<T>::debug_vars(const std::string &fpath, const std::vector<size_t> &qidx,
                            std::vector<size_t> pshape, std::vector<size_t> pstride, size_t pnum,
                            std::vector<size_t> padding, std::vector<size_t> pnum_offset) {
    set_init_vars(fpath, qidx, pshape, pstride, padding, pnum_offset);
    open_file();
    set_runtime_vars(pnum);
    move_stream_to_start();
    sanity_check();
    has_run = true;
}

#endif  // PATCHER_HPP_
