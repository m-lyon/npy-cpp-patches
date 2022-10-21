// Copyright (c) 2022 Matthew Lyon. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef NPY_HEADER_HPP_
#define NPY_HEADER_HPP_

#include <iostream>  // std::cout
#include <array> // std::array
#include <cstring>  // std::size_t
#include <vector>  // std::vector
#include <tuple>  // std::get, std::tie
#include <complex>  // std::complex
#include <algorithm>  // std::all_of
#include <utility>  // std::pair
#include <string>  // std::string

namespace npy_header {

/* Compile-time test for byte order.
If your compiler does not define these per default, you may want to define
one of these constants manually. 
Defaults to little endian order. */
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || \
    defined(__BIG_ENDIAN__) || \
    defined(__ARMEB__) || \
    defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || \
    defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
const bool big_endian = true;
#else
const bool big_endian = false;
#endif

constexpr char magic_string[] = "\x93NUMPY";
constexpr size_t magic_string_length = 6;

constexpr char little_endian_char = '<';
constexpr char big_endian_char = '>';
constexpr char no_endian_char = '|';

constexpr std::array<char, 3> endian_chars = {little_endian_char, big_endian_char, no_endian_char};
constexpr std::array<char, 4> numtype_chars = {'f', 'i', 'u', 'c'};

// determine host endianess
constexpr char host_endian_char = (big_endian ? big_endian_char : little_endian_char);

// npy version
typedef std::pair<char, char> version_t;

struct dtype_t {
    const char byteorder;
    const char kind;
    const unsigned int itemsize;

    std::string str() {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%c%c%u", byteorder, kind, itemsize);
        return std::string(buf);
    }

    std::tuple<const char, const char, const unsigned int> tie() const {
        return std::tie(byteorder, kind, itemsize);
    }
};

struct header_t {
    const dtype_t dtype;
    const bool fortran_order;
    const std::vector<size_t> shape;
};

version_t read_magic(std::istream&);

// --- Begin typestring templates ---

// General template
template<typename T>
struct has_typestring{
    static const bool value = false;
};

// floats specialisations
template<>
struct has_typestring<float>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'f', sizeof(float)};
};

template<>
struct has_typestring<double>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'f', sizeof(double)};
};

template<>
struct has_typestring<long double>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'f', sizeof(long double)};
};

// ints specialisations
template<>
struct has_typestring<int16_t>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'i', sizeof(int16_t)};
};

template<>
struct has_typestring<int>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'i', sizeof(int)};
};

template<>
struct has_typestring<int64_t>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'i', sizeof(int64_t)};
};

// unsigned ints specialisations
template<>
struct has_typestring<unsigned short>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'u', sizeof(unsigned short)};
};

template<>
struct has_typestring<unsigned int>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'u', sizeof(unsigned int)};
};

template<>
struct has_typestring<unsigned long>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'u', sizeof(unsigned long)};
};

// char specialisations
template<>
struct has_typestring<char>{
    static const bool value = true;
    static constexpr dtype_t dtype = {no_endian_char, 'i', sizeof(char)};
};

template<>
struct has_typestring<signed char>{
    static const bool value = true;
    static constexpr dtype_t dtype = {no_endian_char, 'i', sizeof(signed char)};
};

template<>
struct has_typestring<unsigned char>{
    static const bool value = true;
    static constexpr dtype_t dtype = {no_endian_char, 'u', sizeof(unsigned char)};
};

// complex specialisations
template<>
struct has_typestring<std::complex<float>>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'c', sizeof(std::complex<float>)};
};

template<>
struct has_typestring<std::complex<double>>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'c', sizeof(std::complex<double>)};
};

template<>
struct has_typestring<std::complex<long double>>{
    static const bool value = true;
    static constexpr dtype_t dtype = {host_endian_char, 'c', sizeof(std::complex<long double>)};
};

// --- End typestring templates ---

std::string read_header(std::istream&);
header_t parse_header(std::string);
dtype_t parse_descr(std::string);


inline bool is_digits(const std::string &str) {
    // ::isdigit exists in root namespace, explicitly referenced by "::"
    return std::all_of(str.begin(), str.end(), ::isdigit);
}


template<typename T, size_t N>
inline bool in_array(T val, const std::array<T, N>& arr) {
    return std::find(std::begin(arr), std::end(arr), val) != std::end(arr);
}

}  // namespace npy_header

#endif  // NPY_HEADER_HPP_
