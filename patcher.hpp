#ifndef PATCHER_HPP_
#define PATCHER_HPP_

#include "npy.hpp"

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
        std::vector<size_t> num_patches, padding, strides, shifts;
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
        void close_file();

    public:
        std::vector<T> get_patch(
            std::string&, std::vector<size_t>&, std::vector<size_t>, std::vector<size_t>
        );
        size_t get_patch_size();
        std::vector<size_t> get_data_shape();
        std::vector<size_t> get_padding();
};


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

    // Init patch object
    set_patch_size();
    patch.resize(patch_size, 5);
}


/**
 * @brief Opens npy file ready for data extraction. Reads and parses header.
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::open_file(){
    stream.open(filepath, std::ifstream::binary);
    std::string header_s = npy::read_header(stream);
    start = stream.tellg();
    npy::header_t header = npy::parse_header(header_s);
    data_shape = header.shape;
    std::reverse(data_shape.begin(), data_shape.end());
}


/**
 * @brief Closes file after finished extracting patch.
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::close_file(){
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
 * @tparam T 
 * @return std::vector<size_t> Padding
 */
template<typename T>
std::vector<size_t> Patcher<T>::get_padding(){
    std::vector<size_t> out(padding.size());
    std::reverse_copy(padding.begin(), padding.end(), out.begin());
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
    for (auto i: patch_shape){
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
 * @brief Sets strides vector
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_strides(){
    strides.resize(patch_shape.size() + 1, 0);

    strides[0] = sizeof(T); // 0th dimension moves linearly
    for (int i = 1; i <= patch_shape.size(); i++){
        strides[i] = data_shape[i-1] * strides[i-1];
    }

}


/**
 * @brief Set the num_patches object
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_num_patches(){
    num_patches.resize(data_shape.size() - 1);
    for (int i = 0; i < num_patches.size(); i++){
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
    int i = 0;
    pos = 0;
    // get relative position of patched dims
    for (i; i < patch_shape.size(); i++){
        if (patch_num[i] != 0){
            // shift minus the padding
            pos += (strides[i] * patch_num[i] * patch_shape[i]) - (strides[i] * padding[2*i]); 
        }
    }
    pos += (qspace_index[0] * strides[i]); // qdim
    pos += start;
    stream.seekg(start + pos, stream.beg);
}


/**
 * @brief Sets actual byte shift lengths for stream/buffer
 * 
 * @tparam T datatype of data found within filepath
 */
template<typename T>
void Patcher<T>::set_shift_lengths(){
    shifts.resize(patch_shape.size(), 0);

    for (int i = 0; i < shifts.size(); i++){
        shifts[i] = strides[i] * patch_shape[i];
        // If start of patch
        if (patch_num[i] == 0){
            shifts[i] -= strides[i] * padding[2*i];
        // If end of patch
        } else if (patch_num[i] == num_patches[i] - 1){
            shifts[i] -= strides[i] * padding[(2*i)+1];
        }
    }
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
    set_shift_lengths();
    set_num_patches();
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
    buf = reinterpret_cast<char *>(patch.data());
    const unsigned int dim = patch_shape.size();
    for (int i = 0; i < qspace_index.size() - 1; i++){
        read_nd_slice(dim - 1);
        pos -= shifts[dim - 1];
        pos += ((qspace_index[i + 1] - qspace_index[i]) * strides.back());
        stream.seekg(pos, stream.beg);
        std::cout << "we've just read dim " << dim << " index " << i << ", moving stream ptr " << ((qspace_index[i + 1] - qspace_index[i]) * strides.back()) - shifts[dim - 1] << " bytes." << std::endl;
    }
    read_nd_slice(dim - 1); // last slice
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
        if ((patch_num[dim] == 0) && (padding[2 * dim] > 0)){
            std::cout << "we're at the left padded region in dim " << dim << ", shifting data ptr forward " << strides[dim] * padding[2 * dim] << " bytes." << std::endl;
            buf += strides[dim] * padding[2 * dim];
        }
        // Read and shift pointers
        stream.read(buf, shifts[0]);
        buf += shifts[0];
        printf("Address of x is %p\n", (void *)buf);
        pos += shifts[0];
        std::cout << "after reading " << shifts[0] << " bytes, moving data ptr and (implicitly) stream forward " << shifts[0] << " bytes." << std::endl;
        if ((patch_num[dim] == num_patches[dim]) && (padding[(2 * dim) + 1] > 0)){
            std::cout << "we're at the right padded region in dim " << dim << ", moving data ptr forward " << shifts[dim] << " bytes." << std::endl;
            buf += shifts[dim];
        }
        return;
    }
    if (dim < patch_shape.size()){
        for (int i = 0; i < (patch_shape[dim]); i++){
            // If at first patch, and i is within left padded region
            if ((patch_num[dim] == 0) && (i < padding[2 * dim])){
                std::cout << "we're at the left padded region in dim " << dim << ", shifting data ptr forward " << strides[dim] * padding[2 * dim] << " bytes." << std::endl;
                buf += strides[dim] * padding[2 * dim];
            // If at end patch, and i is within right padded region
            } else if ((patch_num[dim] == num_patches[dim]) && (i >= patch_shape[dim] - padding[(2 * dim) + 1] - 1)){
                std::cout << "we're at the right padded region in dim " << dim << ", moving data ptr forward " << strides[dim] * padding[(2 * dim) + 1] << " bytes." << std::endl;
                buf += strides[dim] * padding[(2 * dim) + 1];
            } else {
                read_nd_slice(dim - 1);
                pos = pos - shifts[dim - 1] + strides[dim]; // Shift stream position.
                stream.seekg(pos, stream.beg);
                std::cout << "we've just read dim " << dim << " index " << i << ", moving stream ptr forward " << strides[dim] - shifts[dim - 1] << " bytes." << std::endl;
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
    close_file();

    return patch;
}


#endif  // PATCHER_HPP_