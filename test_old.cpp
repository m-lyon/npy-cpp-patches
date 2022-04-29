
#include "npy_data_padded.cpp"
#include <vector>
#include <string>

using namespace std;



int main(){

    string fpath = "data2.npy";
    vector<size_t> qspace_index {0, 1};
    vector<size_t> patch_shape {2, 2};
    vector<size_t> patch_num {0, 0};

    vector<long> patch = patcher::get_patch_c_order<long>(
        fpath, qspace_index, patch_shape, patch_num
    );

    // string fpath = "data3.npy";
    // vector<size_t> data_shape {2, 2};
    // vector<size_t> qspace_index {0, 1};
    // vector<size_t> patch_shape {2};
    // vector<size_t> patch_num {0};
    // vector<size_t> padding {0, 0};

    // string fpath = "data4.npy";
    // vector<size_t> data_shape {4, 6, 6};
    // vector<size_t> qspace_index {1, 3};
    // vector<size_t> patch_shape {2, 2};
    // vector<size_t> patch_num {0, 0};
    // vector<size_t> padding {0, 0, 0, 0};

    patcher::print_vector(patch);


    return 0;
}