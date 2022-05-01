'''Testing Patcher class'''
import os
import unittest
import numpy as np

from example import PatcherDouble, PatcherFloat, PatcherInt, PatcherLong


def get_test_data_one(filepath):
    '''Testing: Simple test, shame input and output data

    Datatype: double
    Padding required: None
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.double)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.arange(9).reshape(1, 3, 3).astype(np.double)
    data_dict = {
        'patch_shape': (3, 3),
        'qdx': (0,),
        'patch_num': (0, 0),
        'padding': (0, 0, 0, 0),
        'data_out': data_out,
    }

    return data_dict


def get_test_data_two(filepath):
    '''Testing: padding required, getting patch at one edge

    Datatype: float
    Padding required: (1, 0, 1, 0)
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.float32)
    data_in = np.pad(data_in, ((0, 0), (1, 1), (1, 1)), constant_values=42)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.array([[[0, 0, 0], [42, 42, 42], [1, 2, 42]]]).astype(np.float32)
    data_dict = {
        'patch_shape': (3, 3),
        'qdx': [0],
        'patch_num': (0, 1),
        'padding': (1, 0, 1, 0),
        'data_out': data_out,
    }

    return data_dict


def get_test_data_three(filepath):
    '''Testing: more complex qspace indexing

    Datatype: int
    Padding required: (1, 0, 1, 0)
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.int32)
    data_in = np.pad(data_in, ((0, 0), (1, 1), (1, 1)), constant_values=42)
    rand = lambda x: np.random.randint(0, 50, (x, 5, 5), dtype=np.int32)
    data_in = np.concatenate([data_in, rand(4), data_in, data_in, rand(2)], axis=0)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.array([[[0, 42, 3], [0, 42, 6], [0, 42, 42]]]).astype(np.int32)
    data_out = np.concatenate([data_out, data_out, data_out], axis=0)
    data_dict = {
        'patch_shape': [3, 3],
        'qdx': np.array([0, 5, 6]),
        'patch_num': (1, 0),
        'padding': (1, 0, 1, 0),
        'data_out': data_out,
    }
    
    return data_dict


def get_test_data_four(filepath):
    '''Testing: more complex (3D) shapes, differing qspace indexing
    
    Datatype: long
    Padding required: (0, 0, 4, 3, 2, 1)
    '''
    data_in = np.arange(12*33*22).reshape(1, 12, 33, 22).astype(np.int64)
    rand = lambda x: np.random.randint(0, 300, (x, 12, 33, 22), dtype=np.int64)
    data_in_final = np.concatenate([data_in, rand(1), data_in*2, rand(3), data_in*3], axis=0)
    np.save(filepath, data_in_final, allow_pickle=False)
    data_out = np.pad(data_in_final, ((0, 0), (0, 0), (4, 3), (2, 1)))[:, 6:9, 30:40, 0:5]
    data_dict = {
        'patch_shape': (3, 10, 5),
        'qdx': np.array([6, 0, 2]),
        'patch_num': (2, 3, 0),
        'padding': (0, 0, 4, 3, 2, 1),
        'data_out': data_out,
        'data_strides': (8*22*33*12, 8*22*33, 8*22, 8),
        'patch_strides': (10*5*8, 5*8, 8),
        'shift_lengths': (3*8*22*33, 7*8*22, 3*8),
        'stream_start': (57200*8) + 128,
    }

    return data_dict


def get_test_data_five(filepath):
    '''Testing: Patch shape larger than data shape

    Datatype: float
    Padding required: (1, 1, 2, 1, 0, 0, 1, 0)
    '''
    data_in = np.arange(4*7*20*3).reshape(1, 4, 7, 20, 5).astype(np.float32)
    rand = lambda x: np.random.randn(x, 4, 7, 20, 5).astype(np.float32)
    data_in_final = np.concatenate([data_in, rand(4), data_in*0.7, rand(3)], axis=0)
    np.save(filepath, data_in_final, allow_pickle=False)
    data_out = data_in[:, :, :, 10:15, 0:2]
    data_out = np.pad(data_out, ((0, 0), (1, 1), (2, 1), (0, 0), (1, 0)), constant_values=0)
    data_out = np.concatenate([data_out*0.7, data_out], axis=0)
    data_dict = {
        'patch_shape': (6, 10, 5, 3),
        'qdx': np.array([5, 0]),
        'patch_num': (0, 0, 2, 0),
        'padding': (1, 1, 2, 1, 0, 0, 1, 0),
        'data_out': data_out,
    }

    return data_dict

# TODO: test last patch in file to see if errors
# TODO: test realistic size dataset

class BaseTestCases:
    
    class BaseTest(unittest.TestCase):
        # pylint: disable=no-member,attribute-defined-outside-init

        def setUp(self) -> None:
            self.set_up_vars()
            self.data_dict = self.setup_func(self.fpath)

        def tearDown(self):
            os.remove(self.fpath)

        def set_up_vars(self):
            raise NotImplementedError

        def run_get_patch(self):
            return self.patcher.get_patch(
                self.fpath, self.data_dict['qdx'], self.data_dict['patch_shape'], self.data_dict['patch_num']
            )

        def test_equality(self):
            data_out_test = self.run_get_patch()
            patch_shape = (len(self.data_dict['qdx']), ) + tuple(self.data_dict['patch_shape'])
            data_out_test = np.array(data_out_test).reshape(patch_shape)
            print("\n\n")
            print(data_out_test[0, 0, 0])
            print("\n")
            print(self.data_dict['data_out'][0, 0, 0])
            print("\n\n")
            self.assertTrue(np.array_equal(data_out_test, self.data_dict['data_out']))

        def test_padding(self):
            _ = self.run_get_patch()
            padding = tuple(self.patcher.get_padding())
            self.assertEqual(padding, self.data_dict['padding'])


class ExtraTestMixin:

    def test_data_strides(self):
        _ = self.run_get_patch()
        data_strides = tuple(self.patcher.get_data_strides())
        self.assertEqual(data_strides, self.data_dict['data_strides'])

    def test_patch_strides(self):
        _ = self.run_get_patch()
        patch_strides = tuple(self.patcher.get_patch_strides())
        self.assertEqual(patch_strides, self.data_dict['patch_strides'])

    def test_shifts(self):
        _ = self.run_get_patch()
        shifts = tuple(self.patcher.get_shift_lengths())
        self.assertEqual(shifts, self.data_dict['shift_lengths'])

    def test_stream_start(self):
        _ = self.run_get_patch()
        start = self.patcher.get_stream_start()
        self.assertEqual(start, self.data_dict['stream_start'])


# class TestPatcherOne(BaseTestCases.BaseTest):

#     def set_up_vars(self):
#         self.fpath = 'test_data_one.npy'
#         self.setup_func = get_test_data_one
#         self.patcher = PatcherDouble()


# class TestPatcherTwo(BaseTestCases.BaseTest):

#     def set_up_vars(self):
#         self.fpath = 'test_data_two.npy'
#         self.setup_func = get_test_data_two
#         self.patcher = PatcherFloat()


# class TestPatcherThree(BaseTestCases.BaseTest):

#     def set_up_vars(self):
#         self.fpath = 'test_data_three.npy'
#         self.setup_func = get_test_data_three
#         self.patcher = PatcherInt()


class TestPatcherFour(BaseTestCases.BaseTest, ExtraTestMixin):

    def set_up_vars(self):
        self.fpath = 'test_data_four.npy'
        self.setup_func = get_test_data_four
        self.patcher = PatcherLong()


if __name__ == '__main__':
    unittest.main()
