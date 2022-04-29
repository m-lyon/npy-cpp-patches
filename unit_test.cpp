#include "npy_data_padded.cpp"
#include <vector>
#include <string>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("This is a test indeed", "[single-file]") {

    std::vector<size_t> data_shape {2, 2, 2};
    std::vector<size_t> patch_shape {2, 2};

    REQUIRE( patcher::get_strides(data_shape, patch_shape, sizeof(float))[0] == 8);

    // get_strides(data_shape, patch_shape, sizeof(T))
}