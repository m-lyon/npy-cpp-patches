#include <iostream> // std::cout
#include <cstring> // std::size_t
#include <vector>
#include <tuple> // std::get, std::tie
// #include <array> // 

namespace npy {

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

    const char magic_string[] = "\x93NUMPY";
    const std::size_t magic_string_length = 6;

    const char little_endian_char = '<';
    const char big_endian_char = '>';
    const char no_endian_char = '|';

    constexpr char endian_chars[] = {little_endian_char, big_endian_char, no_endian_char};
    constexpr char numtype_chars[] = {'f', 'i', 'u', 'c'};
    // determine host endianess
    constexpr char host_endian_char = (big_endian ? big_endian_char : little_endian_char);

    // npy array length
    typedef unsigned long int ndarray_len_t;
    typedef std::pair<char, char> version_t;

    struct dtype_t {
        const char byteorder;
        const char kind;
        const unsigned int itemsize;

        std::string str() {
            const std::size_t max_buflen = 16;
            char buf[max_buflen];
            std::snprintf(buf, max_buflen, "%c%c%u", byteorder, kind, itemsize);
            return std::string(buf);
        }

        std::tuple<const char, const char, const unsigned int> tie() {
            return std::tie(byteorder, kind, itemsize);
        }
    };

    struct header_t {
        const dtype_t dtype;
        const bool fortran_order;
        const std::vector<ndarray_len_t> shape;
    };

    // This was an inline function before but
    version_t read_magic(std::istream &istream) {
        char buf[magic_string_length + 2];
        istream.read(buf, magic_string_length + 2);

        if (!istream) {
            throw std::runtime_error("io error: failed reading file");
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
    };

    // Begin typestring templates

    // General template
    template<typename T>
    struct has_typestring {
        static const bool value = false;
    };

    // float specialisation
    template<>
    struct has_typestring<float> {
        static const bool value = true;
        static constexpr dtype_t
        dtype = {host_endian_char, 'f', sizeof(float)};
    };

    // NOTE: here I am confused
    constexpr dtype_t
    has_typestring<float>::dtype;
    template<>
    struct has_typestring<double> {
        static const bool value = true;
        static constexpr dtype_t
        dtype = {host_endian_char, 'f', sizeof(double)};
    };
}

int main(){

    npy::dtype_t d = {'>', 'i', 4};
    std::cout << std::get<1>(d.tie()) << std::endl;
    return 0;
}

/* Questions:

    1. What do inline functions do, when to use/when not to use
    2. Presumably you need a general template before you define specialised templates?
    3. The templating of has_typestring seems very verbose and repeated... why do you think
        the author wrote it this way? is there a cleaner way to write this?
    4. When using objects/functions within the std namespace, is it best practice to always write
        std::your_method, or to just declare using namespace std at the top?
    5. Is there any best practice for which case you use, e.g. camelCase vs PascalCase vs snake_case?
    6. Best practice for long amount of func args? use a struct instead?
    7. Linters?
*/