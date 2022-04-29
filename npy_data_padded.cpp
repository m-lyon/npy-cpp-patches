#include "npy.hpp"

using namespace std;

/*
    Restrictions:
        1st dimension is q-space index
        2nd+ dimensions are patches (can be 1+D, not just 3D)
        dimension of data == dimension of patches + 1

*/



namespace patcher {

    template<typename T>
    void print_vector(vector<T> &data){
        for (auto i: data){
            cout << i << ',';
        }
        cout << endl;
    }

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
     * @brief Get the data strides for each dimension
     * 
     * @param data_shape The shape of the underlying data.
     * @param patch_shape The shape of the patch to extract.
     * @param data_size Size in bytes of data type. e.g. 4 for float.
     * @return vector<size_t> The strides vector
     */
    vector<size_t> get_strides(vector<size_t> &data_shape, vector<size_t> &patch_shape, size_t data_size){
        vector<size_t> strides (patch_shape.size() + 1);

        strides[0] = data_size; // 0th dimension moves linearly
        for (int i = 1; i <= patch_shape.size(); i++){
            strides[i] = data_shape[i-1] * strides[i-1];
        }

        return strides;
    }

    vector<size_t> get_num_patches(vector<size_t>& data_shape, vector<size_t>& patch_shape, vector<size_t>& padding){
        vector<size_t> num_patches (data_shape.size() - 1);
        for (int i = 0; i < num_patches.size(); i++){
            num_patches[i] = (data_shape[i] + padding[2*i] + padding[(2*i)+1]) / patch_shape[i];
        }

        return num_patches;
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
    size_t move_stream_to_start_of_patch(
        ifstream& stream,
        vector<size_t>& strides,
        vector<size_t>& patch_shape,
        vector<size_t>& patch_num,
        vector<size_t>& qspace_index,
        vector<size_t>& padding,
        const int& start
    ){
        size_t pos = 0;
        int i = 0;
        // get relative position of patched dims
        for (i; i < patch_shape.size(); i++){
            if (patch_num[i] != 0){
                // shift minus the padding
                pos += (strides[i] * patch_num[i] * patch_shape[i]) - (strides[i] * padding[2*i]); 
            }
        }
        pos += (qspace_index[0] * strides[i]); // qdim

        stream.seekg(start + pos, stream.beg);

        return start + pos;
    }

    void move_data_ptr_to_patch_start(
        char *&buf,
        vector<size_t>& patch_num,
        vector<size_t>& strides,
        vector<size_t>& padding
    ){
        for (int i = 0; i < patch_num.size(); i++){
            if (patch_num[i] == 0){
                cout << "moving data ptr " << strides[i] * padding[2 * i] << endl;
                buf += strides[i] * padding[2 * i];
            }
        }
    }

    vector<size_t> get_shift_lengths(
        vector<size_t>& strides,
        vector<size_t>& patch_shape,
        vector<size_t>& padding,
        vector<size_t>& patch_num,
        vector<size_t>& num_patches
    ){
        vector<size_t> shifts (patch_shape.size());
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
        return shifts;
    }

    /**
     * @brief Loads 1D slice from stream
     * 
     * @param stream 
     * @param buf 
     * @param strides 
     * @param patch_shape 
     * @param dim 
     * @param pos 
     */
    void load_slice(
        ifstream& stream,
        char*& buf,
        vector<size_t>& strides,
        vector<size_t>& patch_shape,
        unsigned int dim,
        size_t& pos,
        vector<size_t>& patch_num,
        vector<size_t>& padding,
        vector<size_t>& shifts,
        vector<size_t>& num_patches
    ){
        if (dim == 0){
            if ((patch_num[dim] == 0) && (padding[2 * dim] > 0)){
                cout << "we're at the left padded region in dim " << dim << ", shifting data ptr forward " << strides[dim] * padding[2 * dim] << " bytes." << endl;
                buf += strides[dim] * padding[2 * dim];
            }
            // Read and shift pointers
            stream.read(buf, shifts[0]);
            buf += shifts[0];
            printf("Address of x is %p\n", (void *)buf);
            pos += shifts[0];
            cout << "after reading " << shifts[0] << " bytes, moving data ptr and (implicitly) stream forward " << shifts[0] << " bytes." << endl;
            if ((patch_num[dim] == num_patches[dim]) && (padding[(2 * dim) + 1] > 0)){
                cout << "we're at the right padded region in dim " << dim << ", moving data ptr forward " << shifts[dim] << " bytes." << endl;
                buf += shifts[dim];
            }
            return;
        }
        if (dim < patch_shape.size()){
            for (int i = 0; i < (patch_shape[dim]); i++){
                // If at first patch, and i is within left padded region
                if ((patch_num[dim] == 0) && (i < padding[2 * dim])){
                    cout << "we're at the left padded region in dim " << dim << ", shifting data ptr forward " << strides[dim] * padding[2 * dim] << " bytes." << endl;
                    buf += strides[dim] * padding[2 * dim];
                    // don't do anything as data ptr has already been moved
                // If at end patch, and i is within right padded region
                } else if ((patch_num[dim] == num_patches[dim]) && (i >= patch_shape[dim] - padding[(2 * dim) + 1] - 1)){
                    cout << "we're at the right padded region in dim " << dim << ", moving data ptr forward " << strides[dim] * padding[(2 * dim) + 1] << " bytes." << endl;
                    buf += strides[dim] * padding[(2 * dim) + 1];
                } else {
                    load_slice(stream, buf, strides, patch_shape, dim - 1, pos, patch_num, padding, shifts, num_patches);
                    pos = pos - shifts[dim - 1] + strides[dim]; // Shift stream position.
                    stream.seekg(pos, stream.beg);
                    cout << "we've just read dim " << dim << " index " << i << ", moving stream ptr forward " << strides[dim] - shifts[dim - 1] << " bytes." << endl;

                    // if (patch_num[dim - 1] == 0){
                    //     cout << "we're at the 0th patch in dim " << dim - 1 << ", shifting data ptr forward " << strides[dim - 1] * padding[2 * (dim - 1)] << " bytes." << endl;
                    //     buf += strides[dim - 1] * padding[2 * (dim - 1)];
                    // }
                    // else if (patch_num[dim - 1] == num_patches[dim - 1] - 1){
                    //     cout << "we're at the " << patch_num[dim - 1] << "th patch in dim " << dim - 1 << ", shifting data ptr forward " << strides[dim - 1] * padding[(2 * (dim - 1)) + 1] << " bytes." << endl;
                    //     buf += strides[dim - 1] * padding[(2 * (dim - 1)) + 1];
                    // }
                }
            }
        }
    }

    /**
     * @brief Loads patch 1D slice at a time from stream
     * 
     * @tparam T 
     * @param data 
     * @param stream 
     * @param strides 
     * @param patch_shape 
     * @param qspace_index 
     * @param pos 
     */
    template<typename T>
    void load_patch(
        vector<T> &data,
        ifstream &stream,
        vector<size_t> &strides,
        vector<size_t> &patch_shape,
        vector<size_t> &qspace_index,
        size_t &pos,
        vector<size_t>& patch_num,
        vector<size_t>& padding,
        vector<size_t>& shifts,
        vector<size_t>& num_patches
    ){

        // get data pointer as char pointer
        char* buf = reinterpret_cast<char *>(data.data());
        const int dim = patch_shape.size();
        for (int i = 0; i < qspace_index.size() - 1; i++){
            load_slice(stream, buf, strides, patch_shape, dim - 1, pos, patch_num, padding, shifts, num_patches);
            pos -= shifts[dim - 1];
            pos += ((qspace_index[i + 1] - qspace_index[i]) * strides.back());
            cout << "we've just read dim " << dim << " index " << i << ", moving stream ptr " << ((qspace_index[i + 1] - qspace_index[i]) * strides.back()) - shifts[dim - 1] << " bytes." << endl;
            stream.seekg(pos, stream.beg);
        }
        load_slice(stream, buf, strides, patch_shape, dim - 1, pos, patch_num, padding, shifts, num_patches); // last slice
    }

    ifstream open_file(string& fpath){
    // ifstream open_file(string& fpath){
        ifstream stream;
        stream.open(fpath, ifstream::binary);
        return stream;
        // return stream;
    }

    vector<size_t> get_data_shape(ifstream& stream){
        vector<size_t> data_shape;
        string header_s = npy::read_header(stream);
        npy::header_t header = npy::parse_header(header_s);
        data_shape = header.shape;

        return data_shape;
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
    vector<T> get_patch_c_order(
        string &filepath,
        vector<size_t> &qspace_index,
        vector<size_t> &patch_shape,
        vector<size_t> &patch_num
    ){

        ifstream stream = open_file(filepath);
        vector<size_t> data_shape = get_data_shape(stream);

        vector<size_t> padding = get_padding(data_shape, patch_shape);

        const int start = stream.tellg();

        // Instantiate patch vector
        vector<T> data (total_size(patch_shape) * qspace_index.size());

        // Reverse dimension vectors
        reverse(padding.begin(), padding.end());
        reverse(patch_shape.begin(), patch_shape.end());
        reverse(data_shape.begin(), data_shape.end());
        reverse(patch_num.begin(), patch_num.end());

        // Initialise strides vector
        vector<size_t> strides = get_strides(data_shape, patch_shape, sizeof(T));

        // Get number of patches
        vector<size_t> num_patches = get_num_patches(data_shape, patch_shape, padding);

        // // Move to start of patch
        size_t pos = move_stream_to_start_of_patch(stream, strides, patch_shape, patch_num, qspace_index, padding, start);

        // // Get buffer shifts for each dimension
        vector<size_t> shifts = get_shift_lengths(strides, patch_shape, padding, patch_num, num_patches);

        // // load_patch
        load_patch(data, stream, strides, patch_shape, qspace_index, pos, patch_num, padding, shifts, num_patches);
        // cout << &data << endl;
        // cout << &data[0] << endl;

        stream.close();
        return data;
    }
}
