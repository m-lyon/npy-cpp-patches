#include "patcher.hpp"
#include <vector>
#include <string>

using namespace std;


template<typename T>
void print_vector(vector<T> &data){
    for (auto i: data){
        cout << i << ',';
    }
    cout << endl;
}


int main(){

    string fpath = "test_data_two.npy";
    vector<size_t> qspace_index {0};
    vector<size_t> patch_shape {3, 3};
    vector<size_t> patch_num {0, 1};

    Patcher<float> patcher;

    vector<float> patch = patcher.get_patch(fpath, qspace_index, patch_shape, patch_num);

    vector<size_t> ans = patcher.get_data_shape();
    size_t total_size = patcher.get_patch_size();
    cout << total_size << endl;
    print_vector(patch);
    print_vector(ans);


    return 0;
}