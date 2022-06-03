# npy-cpp-patches
Read N-Dimensional patches from `.npy` files. This module is built using `C++` and has `Python3` bindings.

## Data Specifications

- Arrays must be saved in `C-contiguous` format, i.e. **NOT** `Fortran-contiguous`.
- First dimension is indexed using in a non-contiguous manner. For example, this can be used to extract specific channels within a natural image.
- Next dimensions are specified by a patch shape `C++` vector or `Python` tuple. To extract patches of lower dimensionality than that of the data, set the corresponding dimensions to `1`.


## Python Usage

### Install
```pip install npy-patcher```

### Usage
```python

from npy_patcher import PatcherDouble, PatcherFloat, PatcherInt, PatcherLong

# In this example lets say our data has shape (10, 90, 90), and is therefore 3D.
data_fpath = '/my/numpy/file.npy'
patcher = PatcherFloat() # Use PatcherFloat for np.float32 datatype
nc_index = [0, 1, 3, 5, 7] # Non-contiguous index
patch_shape = (30, 30) # Contiguous patch shape.
patch_num = 2
# The patch number indexes the patches (starting from 0). So in our example the index 2 would be equivalent to data[nc_index, 0:30, 60:90]. The variable indexes the patches in C-contiguous manner, i.e. the last dimension has the smallest stride.

patch = patcher.get_patch(data_fpath, nc_index, patch_shape, patch_num)
patch = patch.reshape((5, 30, 30)) # PatcherFloat returns a list, therefore we need to reshape.
```

## C++ Usage

Below is an example written in `C++`, equivalent to the `Python` usage above.

```cpp
// test.cpp
#include "src/patcher.hpp"
#include <vector>
#include <string>

int main() {
    std::string fpath = "data.npy";
    std::vector<size_t> nc_index {0, 1, 3, 5, 7};
    std::vector<size_t> patch_shape {30, 30};
    size_t patch_num = 2;

    Patcher<float> patcher;

    // Here the patch object is a contiguous 1D vector
    std::vector<float> patch = patcher.get_patch(fpath, nc_index, patch_shape, patch_num);

    return 0;
}
```

You can then build the package, for example using `g++`:

```bash
$ cd npy-cpp-patches/
$ g++ -std=c++17 -I ./ -g test.cpp src/npy_header.cpp src/pyparse.cpp -o test
```
