#!/usr/bin/env python3

from example import get_patch_c_order_double
from unit_tests import get_test_data_one

if __name__ == '__main__':
    patch_shape, qspace_index, patch_num, data_out = get_test_data_one('test_data_one.npy')
    val = get_patch_c_order_double('test_data_one.npy', [1, 3, 3], qspace_index, patch_shape, patch_num)

    print(hex(id(val)))
    print(hex(id(val[0])))
