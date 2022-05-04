#ifndef PATCHER_HPP_
#define PATCHER_HPP_

#include <string> // std::string
#include <fstream> // std::ifstream
#include <vector> // std::vector

#include "npy_header.hpp"

// TODO: Rewrite npy header reading code. Include check to make sure data dtypes
// between the header and the used class match.
// TODO: remove debug code


/**
 * @brief Patcher object
 * 
 * @tparam T datatype of data found within fpath
 */
template<typename T>
class Patcher{

    private:
        std::string filepath;
        std::ifstream stream;
        std::vector<T> patch;
        std::vector<size_t> data_shape, qspace_index, patch_shape, patch_num;
        std::vector<size_t> num_patches, padding, data_strides, patch_strides, shifts;
        unsigned int rem, total_pad;
        size_t patch_size, start, pos;
        char* buf;
        void set_init_vars(std::string&, std::vector<size_t>&, std::vector<size_t>&, std::vector<size_t>&);
        void set_runtime_vars();
        void set_patch_size();
        void open_file();
        void set_padding();
        void set_strides();
        void set_shift_lengths();
        void set_num_patches();
        void move_stream_to_start();
        void read_patch();
        void read_nd_slice(const unsigned int);
        void read_slice();
        void sanity_check();
        const bool debug = false;

    public:
        Patcher();
        std::vector<T> get_patch(
            std::string&, std::vector<size_t>&, std::vector<size_t>, std::vector<size_t>
        );
        void debug_vars(std::string&, std::vector<size_t>&, std::vector<size_t>, std::vector<size_t>);
        size_t get_patch_size();
        size_t get_stream_start();
        std::vector<size_t> get_data_shape();
        std::vector<size_t> get_padding();
        std::vector<size_t> get_data_strides();
        std::vector<size_t> get_patch_strides();
        std::vector<size_t> get_shift_lengths();
};


template<typename T>
Patcher<T>::Patcher(){}


/**
 * @brief Sets internal variables, these are set before any file reading is necessary.
 * 
 * @tparam T datatype of data found within fpath
 * @param fpath filepath for .npy data file
 * @param qidx qspace index (0th index in file)
 * @param pshape patch shape
 * @param pnum patch number for each patch dimension
 */
template<typename T>
void Patcher<T>::set_init_vars(
    std::string& fpath,
    std::vector<size_t>& qidx,
    std::vector<size_t>& pshape,
    std::vector<size_t>& pnum
){
    filepath = fpath;
    qspace_index = qidx;
    patch_shape = pshape;
    std::reverse(patch_shape.begin(), patch_shape.end());
    patch_num = pnum;
    std::reverse(patch_num.begin(), patch_num.end());

    // Init/reset patch object
    set_patch_size();
    if (patch.size() > 0){
        patch.clear();
    }
    patch.resize(patch_size, 0);
}


/**
 * @brief Opens npy file ready for data extraction. Reads and parses header.
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::open_file(){
    stream.open(filepath, std::ifstream::binary);
    std::string header_s = npy_header::read_header(stream);
    start = stream.tellg();
    npy_header::header_t header = npy_header::parse_header(header_s);
    data_shape = header.shape;
    std::reverse(data_shape.begin(), data_shape.end());
    if (!stream) {
        throw std::runtime_error("IO Error: failed to open " + filepath);
    }

    static_assert(npy_header::has_typestring<T>::value, "Unrecognised datatype in file.");

    if (header.dtype.tie() != npy_header::has_typestring<T>::dtype.tie()) {
        throw std::runtime_error("Type mismatch between class and file.");
    }
}


/**
 * @brief Closes file after finished extracting patch.
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::sanity_check(){
    if (!stream) {
        throw std::runtime_error("Failed to get patch within " + filepath);
    }
    stream.close();
}



/**
 * @brief Calculates padding needed to split data into patches given by
 *      patch_shape
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_padding(){
    padding.resize(patch_shape.size() * 2, 0);

    // Iterate over dimensions
    for (size_t i = 0; i < patch_shape.size(); i++){
        rem = data_shape[i] % patch_shape[i];
        if (rem != 0){
            total_pad = patch_shape[i] - rem;
            if (total_pad % 2 == 0){
                padding[i*2] = total_pad / 2;  
            } else {
                padding[i*2] = (total_pad / 2) + 1; 
            }
            padding[(i*2) + 1] = total_pad / 2;
        }
    }

}


/**
 * @brief Get the calculated padding.
 * 
 * @tparam T datatype of data found within filepath
 * @return std::vector<size_t> Padding
 */
template<typename T>
std::vector<size_t> Patcher<T>::get_padding(){
    std::vector<size_t> out(padding.size());
    for (size_t i = 0; i < out.size() / 2; i++){
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
template<typename T>
void Patcher<T>::set_patch_size(){
    patch_size = 1;
    for (size_t i: patch_shape){
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
template<typename T>
size_t Patcher<T>::get_patch_size(){
    return patch_size;
}


/**
 * @brief Sets data_strides vector
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_strides(){
    data_strides.resize(patch_shape.size() + 1, 0);
    data_strides[0] = sizeof(T); // 0th dimension moves linearly
    for (size_t i = 1; i <= patch_shape.size(); i++){
        data_strides[i] = data_shape[i-1] * data_strides[i-1];
    }

    patch_strides.resize(patch_shape.size(), 0);
    patch_strides[0] = data_strides[0];
    for (size_t i = 1; i < patch_shape.size(); i++){
        patch_strides[i] = patch_shape[i-1] * patch_strides[i-1];
    }
}


template<typename T>
std::vector<size_t> Patcher<T>::get_data_strides(){
    std::vector<size_t> out(data_strides.size());
    std::reverse_copy(data_strides.begin(), data_strides.end(), out.begin());
    return out;
}


template<typename T>
std::vector<size_t> Patcher<T>::get_patch_strides(){
    std::vector<size_t> out(patch_strides.size());
    std::reverse_copy(patch_strides.begin(), patch_strides.end(), out.begin());
    return out;
}


/**
 * @brief Set the num_patches object
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_num_patches(){
    num_patches.resize(data_shape.size() - 1);
    for (size_t i = 0; i < num_patches.size(); i++){
        num_patches[i] = (data_shape[i] + padding[2*i] + padding[(2*i)+1]) / patch_shape[i];
    }
}

/**
 * @brief Moves stream pointer to start of patch
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::move_stream_to_start(){
    size_t i = 0;
    pos = 0;
    // get relative position of patched dims
    for (;i < patch_shape.size(); i++){
        if (patch_num[i] != 0){
            // shift minus the padding
            pos += (data_strides[i] * patch_num[i] * patch_shape[i]) - (data_strides[i] * padding[2*i]); 
        }
    }
    pos += (qspace_index[0] * data_strides[i]); // qdim
    // if (debug){
    //     std::cout << "shifting stream to start of patch: forward " << pos << " bytes." << std::endl;
    // }
    pos += start;
    start = pos; // update to patch start position
    stream.seekg(pos, stream.beg);
}


template<typename T>
size_t Patcher<T>::get_stream_start(){
    return start;
}


/**
 * @brief Sets actual byte shift lengths for stream/buffer
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_shift_lengths(){
    shifts.resize(patch_shape.size(), 0);

    for (size_t i = 0; i < shifts.size(); i++){
        shifts[i] = data_strides[i] * patch_shape[i];
        // If start of patch
        if (patch_num[i] == 0){
            shifts[i] -= data_strides[i] * padding[2*i];
        // If end of patch
        } if (patch_num[i] == num_patches[i] - 1){
            shifts[i] -= data_strides[i] * padding[(2*i)+1];
        }
    }
}


template<typename T>
std::vector<size_t> Patcher<T>::get_shift_lengths(){
    std::vector<size_t> out(shifts.size());
    std::reverse_copy(shifts.begin(), shifts.end(), out.begin());
    return out;
}


/**
 * @brief Sets variables and data after reading data header
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_runtime_vars(){
    set_padding();
    set_strides();
    set_num_patches();
    set_shift_lengths();
}


/**
 * @brief Reads patch into patch vector
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::read_patch(){
    move_stream_to_start();
    // get data pointer as char pointer
    buf = reinterpret_cast<char*>(patch.data());
    const unsigned int dim = patch_shape.size();
    for (size_t i = 0; i < qspace_index.size() - 1; i++){
        read_nd_slice(dim - 1);
        pos -= shifts[dim - 1];
        pos += ((qspace_index[i + 1] - qspace_index[i]) * data_strides.back());
        stream.seekg(pos, stream.beg);
        // if (debug){
        //     std::cout << "we've just read dim " << dim;
        //     std::cout << " index " << i << ", moving stream ptr ";
        //     std::cout << ((qspace_index[i + 1] - qspace_index[i]) * data_strides.back()) - shifts[dim - 1];
        //     std::cout << " bytes." << std::endl;
        // }
    }
    read_nd_slice(dim - 1); // last slice
}


template<typename T>
void Patcher<T>::read_slice(){
    // If in first patch, and left padded region
    if ((patch_num[0] == 0) && (padding[0] > 0)){
        // if (debug){
        //     std::cout << "we're at the left padded region in dim 0";
        //     std::cout << ", shifting data ptr forward " << patch_strides[0] * padding[0];
        //     std::cout << " bytes." << std::endl;
        // }
        buf += patch_strides[0] * padding[0];
    }
    if (shifts[0] > 0){
    // Read and shift pointers
        stream.read(buf, shifts[0]);
        buf += shifts[0];
        pos += shifts[0];
    }
    // if (debug){
    //     std::cout << "after reading " << shifts[0];
    //     std::cout << " bytes, moving data ptr and (implicitly) stream forward ";
    //     std::cout << shifts[0] << " bytes." << std::endl;
    // }
    // If in last patch, and right padded region
    if ((patch_num[0] + 1 == num_patches[0]) && (padding[(2 * 0) + 1] > 0)){
        // if (debug){
        //     std::cout << "we're at the right padded region in dim 0";
        //     std::cout << ", moving data ptr forward " << shifts[0];
        //     std::cout << " bytes." << std::endl;
        // }
        buf += patch_strides[0] * padding[1];
    }
}


/**
 * @brief Reads N-dimensional slice, intended to be used recursively.
 * 
 * @tparam T datatype of data found within filepath
 * @param dim Dimensionality of slice, starting at 0.
 */
template<typename T>
void Patcher<T>::read_nd_slice(const unsigned int dim){
    if (dim == 0){
        read_slice();
    } else {
        // Iterate over dimension
        for (size_t i = 0; i < (patch_shape[dim]); i++){
            // If at first patch, and within left padded region
            if ((patch_num[dim] == 0) && (i < padding[2 * dim])){
                // if (debug){
                //     std::cout << "we're at the left padded region in dim " << dim;
                //     std::cout << ", shifting data ptr forward " << patch_strides[dim];
                //     std::cout << " bytes." << std::endl;
                // }
                buf += patch_strides[dim];
            // If at end patch, and within right padded region
            } else if ((patch_num[dim] + 1 == num_patches[dim]) && (i >= patch_shape[dim] - padding[(2 * dim) + 1])){
                // if (debug){
                //     std::cout << "we're at the right padded region in dim " << dim;
                //     std::cout << ", moving data ptr forward " << patch_strides[dim];
                //     std::cout << " bytes." << std::endl;
                // }
                buf += patch_strides[dim];
            } else {
                read_nd_slice(dim - 1);
                pos = pos - shifts[dim - 1] + data_strides[dim]; // Shift stream position.
                stream.seekg(pos, stream.beg);
                // if (debug){
                //     std::cout << "we've just read dim " << dim;
                //     std::cout << " index " << i << ", moving stream ptr forward "; 
                //     std::cout << - shifts[dim - 1] + data_strides[dim] << " bytes." << std::endl;
                // }
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
template<typename T>
std::vector<size_t> Patcher<T>::get_data_shape(){
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
 * @param pnum patch number for each patch dimension
 * @return std::vector<T> Patch data
 */
template<typename T>
std::vector<T> Patcher<T>::get_patch(
    std::string& fpath,
    std::vector<size_t>& qidx,
    std::vector<size_t> pshape,
    std::vector<size_t> pnum
){
    set_init_vars(fpath, qidx, pshape, pnum);
    open_file();
    set_runtime_vars();
    read_patch();
    sanity_check();

    return patch;
}

template<typename T>
void Patcher<T>::debug_vars(
    std::string& fpath,
    std::vector<size_t>& qidx,
    std::vector<size_t> pshape,
    std::vector<size_t> pnum
){
    set_init_vars(fpath, qidx, pshape, pnum);
    open_file();
    set_runtime_vars();
    move_stream_to_start();
    sanity_check();
}


#endif  // PATCHER_HPP_