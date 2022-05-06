// Copyright (c) 2022 Matthew Lyon. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include <unordered_map>  // std::unordered_map
#include "src/npy_header.hpp"
#include "src/pyparse.hpp"


namespace npy_header {


/**
 * @brief Reads magic string present at top of npy header, then determines file
 *      version from next 2 bytes.
 * 
 * @param stream File stream object
 * @return version_t File version
 */
version_t read_magic(std::istream& stream) {
    char buf[magic_string_length + 2];
    stream.read(buf, magic_string_length + 2);

    if (stream.fail()) {
        throw std::runtime_error("IO Error: Failed to read file");
    }

    // Compares character arrays, returns non zero if difference
    if (0 != std::memcmp(buf, magic_string, magic_string_length)) {
        throw std::runtime_error("This file does not have a valid npy format.");
    }

    // Assign version numbers
    version_t version;
    version.first = buf[magic_string_length];
    version.second = buf[magic_string_length + 1];

    return version;
}


/**
 * @brief Reads header from stream object into header string
 * 
 * @param stream File stream
 * @return std::string Output header string
 */
std::string read_header(std::istream& stream) {
    // check magic bytes and version number
    version_t version = read_magic(stream);
    uint32_t header_length;

    if (version == version_t{1, 0}) {
        uint8_t header_len_le16[2];
        uint8_t htest[1];
        stream.read(reinterpret_cast<char*>(header_len_le16), 2);
        // Here header_len_le16[0] is shifted 0 to convert to int type,
        // then bitwise OR'd with header_len_le16[1] to convert to same
        // endianness as machine.
        header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);

        if ((magic_string_length + 2 + 2 + header_length) % 64 != 0) {
            throw std::runtime_error("npy file has incorrect header length.");
        }
    } else if (version == version_t{2, 0}) {
        uint8_t header_len_le32[4];
        stream.read(reinterpret_cast<char *>(header_len_le32), 4);
        // Similar operation as in version 1.0, however now header length is little endian
        // unsigned int (4 bytes instead of 2)
        header_length = (header_len_le32[0] << 0) | (header_len_le32[1] << 8)
                        | (header_len_le32[2] << 16) | (header_len_le32[3] << 24);

        if ((magic_string_length + 2 + 4 + header_length) % 64 != 0) {
            throw std::runtime_error("npy file has incorrect header length.");
        }
    } else {
        throw std::runtime_error("Unsupported npy file format version.");
    }

    std::vector<char> buf_v;
    buf_v.reserve(header_length);
    stream.read(buf_v.data(), header_length);
    std::string header(buf_v.data(), header_length);

    return header;
}

/**
 * @brief Parses npy header metadata
 * 
 * @details The header consists of the following:
        - First 6 bytes are a magic string: exactly "x93NUMPY".
        - Next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.
        - Next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00.
        - Next 2 bytes form a little-endian unsigned short int: the length of the
            header data HEADER_LEN.
        - Next HEADER_LEN bytes form the header data describing the array's format.
            It is an ASCII string which contains a Python literal expression of a dictionary.
            It is terminated by a newline ('n') and padded with spaces ('x20') to make the total
            length of the magic string + 4 + HEADER_LEN be evenly divisible by 64 for
            alignment purposes.
    
            The dictionary contains three keys:

            "descr" : dtype.descr
                An object that can be passed as an argument to the numpy.dtype()
                constructor to create the array's dtype.
            "fortran_order" : bool
                Whether the array data is Fortran-contiguous or not. Since Fortran-contiguous
                arrays are a common form of non-C-contiguity, we allow them to be written
                directly to disk for efficiency.
            "shape" : tuple of int
                The shape of the array.

            For repeatability and readability, the dictionary keys are sorted in alphabetic order.

 * @param header header string extracted from npy file.
 * @return header_t Header metadata object.
 */
header_t parse_header(std::string header) {
    // Remove trailing newline
    if (header.back() != '\n')
        throw std::runtime_error("Invalid header");
    header.pop_back();

    // Parse the dictionary string
    std::vector<std::string> keys {"descr", "fortran_order", "shape"};
    std::unordered_map<std::string, std::string> dict = pyparse::parse_dict(header, keys);

    if (dict.size() == 0)
        throw std::runtime_error("Invalid dictionary in header");

    std::string descr_s = dict["descr"];
    std::string fortran_s = dict["fortran_order"];
    std::string shape_s = dict["shape"];

    std::string descr = pyparse::parse_str(descr_s);
    dtype_t dtype = parse_descr(descr);

    // Convert literal Python bool to C++ bool
    bool fortran_order = pyparse::parse_bool(fortran_s);

    // Parse the shape tuple
    std::vector<std::string> shape_v = pyparse::parse_tuple(shape_s);

    // Convert string integers in tuple to C integers
    std::vector <size_t> shape;
    for (auto item : shape_v) {
        size_t dim = static_cast<size_t>(std::stoul(item));
        shape.push_back(dim);
    }

    return {dtype, fortran_order, shape};
}


/**
 * @brief Parses dtype description string into dtype struct
 * 
 * @param typestring dtype description
 * @return dtype_t dtype struct
 */
dtype_t parse_descr(std::string typestring) {
    if (typestring.length() < 3) {
        throw std::runtime_error("Invalid typestring (length).");
    }

    char byteorder_c = typestring.at(0);
    char kind_c = typestring.at(1);
    std::string itemsize_s = typestring.substr(2);

    if (!in_array(byteorder_c, endian_chars)) {
        throw std::runtime_error("Invalid typestring (byteorder).");
    }

    if (!in_array(kind_c, numtype_chars)) {
        throw std::runtime_error("Invalid typestring (kind)");
    }

    if (!is_digits(itemsize_s)) {
        throw std::runtime_error("Invalid typestring (itemsize)");
    }
    unsigned int itemsize = std::stoul(itemsize_s);

    return {byteorder_c, kind_c, itemsize};
}

}  // namespace npy_header
