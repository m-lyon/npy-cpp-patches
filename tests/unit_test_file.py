'''Testing different files read with the same object'''

import os
import unittest
import numpy as np

from npy_patcher import PatcherInt


def get_test_data_one(filepath):
    '''Testing: padding required, getting patch at one edge

    Datatype: float
    Padding required: (1, 0, 1, 0)
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.int32)
    data_in = np.pad(data_in, ((0, 0), (1, 1), (1, 1)), constant_values=42)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.array([[[0, 0, 0], [42, 42, 42], [1, 2, 42]]]).astype(np.int32)
    data_in = {
        'fpath': filepath,
        'pshape': (3, 3),
        'pstride': (3, 3),
        'qidx': [0],
        'pnum': 1,
        'padding': [],
    }
    data_out = {
        'padding': (1, 0, 1, 0),
        'data_out': data_out,
    }

    return data_in, data_out


def get_test_data_two(filepath):
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
    data_in = {
        'fpath': filepath,
        'pshape': [3, 3],
        'pstride': [3, 3],
        'qidx': np.array([0, 5, 6]),
        'pnum': 2,
        'padding': [0, 0, 0, 0],
    }
    data_out = {
        'padding': (1, 0, 1, 0),
        'data_out': data_out,
    }

    return data_in, data_out


class DoubleFileTest(unittest.TestCase):
    '''Double File test class'''

    def setUp(self) -> None:
        self.filepath1 = 'test_data_file_one.npy'
        self.filepath2 = 'test_data_file_two.npy'
        self.data_in1, self.data_out1 = get_test_data_one(self.filepath1)
        self.data_in2, self.data_out2 = get_test_data_two(self.filepath2)
        self.patcher = PatcherInt()

    def tearDown(self):
        os.remove(self.filepath1)
        os.remove(self.filepath2)

    def run_get_patch(self):
        '''Runs patcher get patch given runtime params'''
        self.patcher.get_patch(**self.data_in1)
        return self.patcher.get_patch(**self.data_in2)

    def test_equality_two(self):
        '''Tests equality of array output'''
        data_out_test = self.run_get_patch()
        patch_shape = (len(self.data_in2['qidx']),) + tuple(self.data_in2['pshape'])
        data_out_test = np.array(data_out_test).reshape(patch_shape)
        self.assertTrue(np.array_equal(data_out_test, self.data_out2['data_out']))


if __name__ == '__main__':
    unittest.main()
