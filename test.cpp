
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

    string fpath = "data2.npy";
    vector<size_t> qspace_index {0, 1};
    vector<size_t> patch_shape {2, 2};
    vector<size_t> patch_num {0, 0};

    Patcher<long> patcher;

    vector<long> patch = patcher.get_patch(fpath, qspace_index, patch_shape, patch_num);

    vector<size_t> ans = patcher.get_data_shape();

    print_vector(patch);
    print_vector(ans);


    return 0;
}