
#include "patcher_simple.hpp"
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

    SimplePatcher<long> patcher;

    vector<long> patch = patcher.get_patch(fpath);

    // vector<size_t> ans = patcher.get_data_shape();
    // size_t total_size = patcher.get_patch_size();
    // cout << total_size << endl;
    print_vector(patch);
    // print_vector(ans);


    return 0;
}