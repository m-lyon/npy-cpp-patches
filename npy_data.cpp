#include "npy.hpp"

using namespace std;

/*
    Restrictions:
        1st dimension is q-space index
        2nd+ dimensions are patches (can be 1+D, not just 3D)
        dimension of data == dimension of patches + 1

*/

template<typename T>
void print_vector(vector<T> &data){
    for (auto i: data){
        cout << i << ',';
    }
    cout << endl;
}


namespace patcher {

    size_t total_size(const vector<size_t> &shape) {
        size_t size = 1;
        for (size_t i: shape)
            size *= i;

        return size;
    }

    /**
     * @brief Gets the padding vector given the indices to be patched.
     * 
     * @param data_shape The shape of the data.
     * @param patch_shape The shape of the patch to extract.
     * @return vector<size_t> - The padding vector, double the size of patch_shape.
     */
    vector<size_t> get_padding(vector<size_t> &data_shape, vector<size_t> &patch_shape){

        unsigned int rem, total_pad;
        vector<size_t> padding (patch_shape.size() * 2, 0);

        // Iterate over dimensions
        for (size_t i = 0; i < patch_shape.size(); i++){
            rem = data_shape[i+1] % patch_shape[i];
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

        return padding;
    }

    /**
     * @brief Get the memory strides for each dimension
     * 
     * @param data_shape The shape of the underlying data.
     * @param patch_shape The shape of the patch to extract.
     * @param data_size Size in bytes of data type. e.g. 4 for float.
     * @return vector<size_t> The strides vector
     */
    vector<size_t> get_strides(vector<size_t> &data_shape, vector<size_t> &patch_shape, size_t data_size){
        vector<size_t> strides;
        strides.resize(patch_shape.size() + 1);

        strides[0] = data_size; // 0th dimension moves linearly
        for (int i = 1; i == patch_shape.size(); i++){
            strides[i] = data_shape[i-1] * strides[i-1];
        }

        print_vector(strides);
        return strides;
    }


    /**
     * @brief Moves stream position to start of patch, ready for reading
     * 
     * @param stream Data stream, assumed to be at start of array.
     * @param strides Strides vector
     * @param patch_shape The shape of the patch to extract, reversed.
     * @param patch_num The patch number (in each dimension) to collect, reversed.
     * @param qspace_index Indices in 0th dimension of data to collect.
     * @param start Starting offset position
     */
    size_t move_to_start_of_patch(istream &stream, vector<size_t> &strides, vector<size_t> &patch_shape, vector<size_t> &patch_num, vector<size_t> &qspace_index, const int &start){
        
        size_t pos = 0;
        int i = 0;
        // get relative position of patched dims
        for (i; i < patch_shape.size(); i++){
            pos += (strides[i] * patch_num[i] * patch_shape[i]);
        }
        pos += (qspace_index[0] * strides[i]); // qdim

        stream.seekg(start + pos, stream.beg);

        return start + pos;
    }

    void load_slice(istream &stream, char *&buf, vector<size_t> &strides, vector<size_t> &patch_shape, unsigned int dim, size_t &pos){
        if (dim == 0){
            stream.read(buf, strides[dim] * patch_shape[dim]);
            buf += strides[dim] * patch_shape[dim];
            pos += strides[dim] * patch_shape[dim];
            return;
        }
        if (dim < patch_shape.size()){
            for (int i = 0; i < patch_shape[dim]; i++){
                load_slice(stream, buf, strides, patch_shape, dim - 1, pos);
                pos = pos - (strides[dim - 1] * patch_shape[dim - 1]) + strides[dim];
                stream.seekg(pos, stream.beg);
            }
        }

    }

    template<typename T>
    void load_patch(vector<T> &data, istream &stream, vector<size_t> &strides, vector<size_t> &patch_shape, vector<size_t> &qspace_index, size_t &pos){

        // get data pointer as char pointer
        char* buf = reinterpret_cast<char *>(data.data());
        const int dim = patch_shape.size();
        for (int i = 0; i < qspace_index.size() - 1; i++){
            load_slice(stream, buf, strides, patch_shape, dim - 1, pos);
            pos -= (strides[dim - 1] * patch_shape[dim - 1]);
            pos += ((qspace_index[i + 1] - qspace_index[i]) * strides.back());
        }
        load_slice(stream, buf, strides, patch_shape, dim - 1, pos); // last slice
    }

    /**
     * @brief Loads patch into memory, given a 'C' like memory ordering of the data
     *      (last dimensions vary the quickest)
     * 
     * @tparam T datatype of data. e.g. float
     * @param stream Data stream, assumed to be at start of array.
     * @param data_shape The shape of the underlying data.
     * @param qspace_index indices of 0th dimension in data to collect.
     * @param patch_shape The shape of the patch to extract.
     * @param patch_num The patch number (in each dimension) to collect.
     * @param padding The padding vector.
     * @return vector<T> Patched data
     * 
     * TODO: not ideal passing empty vector.
     */
    template<typename T>
    void get_patch_c_order(istream &stream, vector<T> &data, vector<size_t> &data_shape, vector<size_t> &qspace_index, vector<size_t> &patch_shape, vector<size_t> &patch_num, vector<size_t> &padding){
        
        // TODO: incorporate padding into calculations

        const int start = stream.tellg();

        // Instantiate patch vector
        data.resize(total_size(patch_shape) * qspace_index.size());

        // Reverse dimension vectors
        reverse(patch_shape.begin(), patch_shape.end());
        reverse(data_shape.begin(), data_shape.end());
        reverse(patch_num.begin(), patch_num.end());

        // Initialise strides vector
        auto strides = get_strides(data_shape, patch_shape, sizeof(T));

        // Move to start of patch
        auto pos = move_to_start_of_patch(stream, strides, patch_shape, patch_num, qspace_index, start);

        // load_patch
        load_patch(data, stream, strides, patch_shape, qspace_index, pos);
    }

}
