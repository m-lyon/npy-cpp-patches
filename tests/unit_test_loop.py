'''Testing Patcher class'''

import os
import unittest
import numpy as np

from npy_patcher import PatcherFloat, PatcherInt, PatcherLong


def get_test_data_2d(filepath):
    '''Testing: 2D shape with non-contiguous qspace indexing

    Datatype: int
    Padding required: (1, 0, 1, 0)
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.int32)
    data_in = np.pad(data_in, ((0, 0), (1, 1), (1, 1)), constant_values=42)
    rand = lambda x: np.random.randint(0, 50, (x, 5, 5), dtype=np.int32)
    data_in = np.concatenate([data_in, rand(4), data_in, data_in, rand(2)], axis=0)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.pad(data_in, ((0, 0), (1, 0), (1, 0)))
    data_out_final = np.stack([data_out[0], data_out[5], data_out[6]])
    data_in = {
        'fpath': filepath,
        'pshape': [3, 3],
        'pstride': [3, 3],
        'qidx': np.array([0, 5, 6]),
        'padding': [],
    }
    data_out = {
        'max_patch_num': (1, 1),
        'padding': (1, 0, 1, 0),
        'data_out': data_out_final,
    }

    return data_in, data_out


def get_test_data_3d(filepath):
    '''Testing: more complex (3D) shapes, differing qspace indexing

    Datatype: long
    Padding required: (0, 0, 4, 3, 2, 1)
    '''
    data_in = np.arange(12 * 33 * 22).reshape(1, 12, 33, 22).astype(np.int64)
    rand = lambda x: np.random.randint(0, 300, (x, 12, 33, 22), dtype=np.int64)
    data_in_final = np.concatenate([data_in, rand(1), data_in * 2, rand(3), data_in * 3], axis=0)
    np.save(filepath, data_in_final, allow_pickle=False)
    data_out = np.pad(data_in_final, ((0, 0), (0, 0), (4, 3), (2, 1)))
    data_out_final = np.stack([data_out[6], data_out[0], data_out[2]], axis=0)
    data_in = {
        'fpath': filepath,
        'qidx': np.array([6, 0, 2]),
        'pshape': (3, 10, 5),
        'pstride': (3, 10, 5),
        'padding': [],
    }
    data_out = {
        'max_patch_num': (3, 3, 4),
        'data_out': data_out_final,
        'padding': (0, 0, 4, 3, 2, 1),
    }

    return data_in, data_out


def get_test_data_4d(filepath):
    '''Testing: Patch shape larger than data shape

    Datatype: float
    Padding required: (1, 1, 2, 1, 0, 0, 1, 0)
    '''
    data_in = np.arange(4 * 7 * 20 * 5).reshape(1, 4, 7, 20, 5).astype(np.float32)
    rand = lambda x: np.random.randn(x, 4, 7, 20, 5).astype(np.float32)
    data_in_final = np.concatenate([data_in, rand(4), data_in * 0.7, rand(3)], axis=0)
    np.save(filepath, data_in_final, allow_pickle=False)
    data_out = np.pad(data_in_final, ((0, 0), (1, 1), (2, 1), (0, 0), (1, 0)))
    data_out_final = np.stack([data_out[5], data_out[0]])
    data_in = {
        'fpath': filepath,
        'pshape': (6, 10, 5, 3),
        'pstride': (6, 10, 5, 3),
        'qidx': np.array([5, 0]),
        'padding': [],
    }
    data_out = {
        'max_patch_num': (0, 0, 3, 1),
        'data_out': data_out_final,
        'padding': (1, 1, 2, 1, 0, 0, 1, 0),
    }

    return data_in, data_out


class BaseTestCases:
    '''Base test case class with TestClass members'''

    class BaseTest(unittest.TestCase):
        '''Actual Base test class'''

        # pylint: disable=no-member

        def setUp(self) -> None:
            self.set_up_vars()
            self.data_in, self.data_out = self.setup_func(self.filepath)

        def tearDown(self):
            os.remove(self.filepath)

        def set_up_vars(self):
            '''Method to setup vars for testing'''
            raise NotImplementedError

        def run_get_patch(self, pnum):
            '''Runs patcher get patch given runtime params'''
            return self.patcher.get_patch(pnum=pnum, **self.data_in)


class TestPatcherLoop2D(BaseTestCases.BaseTest):
    '''2D test case testing each patch'''

    def set_up_vars(self):
        self.filepath = 'test_data_loop_2D.npy'
        self.setup_func = get_test_data_2d
        self.patcher = PatcherInt()

    def test_equality_loop(self):
        '''Tests equality of array output for each patch'''
        max_patch_num = self.data_out['max_patch_num']
        patch_shape = self.data_in['pshape']
        pshape = (len(self.data_in['qidx']),) + tuple(patch_shape)
        pnum = 0
        for i in range(0, max_patch_num[0] + 1):
            for j in range(0, max_patch_num[1] + 1):
                slice_str = '[:, {}:{}, {}:{}]'.format(
                    i * patch_shape[0],
                    (i + 1) * patch_shape[0],
                    j * patch_shape[1],
                    (j + 1) * patch_shape[1],
                )
                with self.subTest(f"Patch: {pnum} data{slice_str}"):
                    data_out_test = self.run_get_patch(pnum)
                    pnums = self.patcher.get_patch_numbers()
                    self.assertEqual([i, j], pnums)
                    data_out_test = np.array(data_out_test).reshape(pshape)
                    data_out_true = self.data_out['data_out'][
                        :,
                        i * patch_shape[0] : (i + 1) * patch_shape[0],
                        j * patch_shape[1] : (j + 1) * patch_shape[1],
                    ]
                    self.assertTrue(
                        np.array_equal(data_out_test, data_out_true),
                        f'\n{data_out_test}\n\n-------\n{data_out_true}',
                    )
                pnum += 1


class TestPatcherLoop3D(BaseTestCases.BaseTest):
    '''3D test case testing each patch'''

    def set_up_vars(self):
        self.filepath = 'test_data_loop.npy'
        self.setup_func = get_test_data_3d
        self.patcher = PatcherLong()

    def test_equality_loop(self):
        '''Tests equality of array output for each patch'''
        max_patch_num = self.data_out['max_patch_num']
        patch_shape = self.data_in['pshape']
        pshape = (len(self.data_in['qidx']),) + tuple(patch_shape)
        pnum = 0
        for i in range(0, max_patch_num[0] + 1):
            for j in range(0, max_patch_num[1] + 1):
                for k in range(0, max_patch_num[2] + 1):
                    slice_str = '[:, {}:{}, {}:{}, {}:{}]'.format(
                        i * patch_shape[0],
                        (i + 1) * patch_shape[0],
                        j * patch_shape[1],
                        (j + 1) * patch_shape[1],
                        k * patch_shape[2],
                        (k + 1) * patch_shape[2],
                    )
                    with self.subTest(f"Patch: ({i},{j},{k}) data{slice_str}"):
                        data_out_test = self.run_get_patch(pnum)
                        data_out_test = np.array(data_out_test).reshape(pshape)
                        data_out_true = self.data_out['data_out'][
                            :,
                            i * patch_shape[0] : (i + 1) * patch_shape[0],
                            j * patch_shape[1] : (j + 1) * patch_shape[1],
                            k * patch_shape[2] : (k + 1) * patch_shape[2],
                        ]
                        self.assertTrue(
                            np.array_equal(data_out_test, data_out_true),
                        )
                    pnum += 1


class TestPatcherLoop4D(BaseTestCases.BaseTest):
    '''4D test case testing each patch'''

    def set_up_vars(self):
        self.filepath = 'test_data_loop_4D.npy'
        self.setup_func = get_test_data_4d
        self.patcher = PatcherFloat()

    def test_equality_loop(self):
        '''Tests equality of array output for each patch'''
        max_patch_num = self.data_out['max_patch_num']
        patch_shape = self.data_in['pshape']
        pshape = (len(self.data_in['qidx']),) + tuple(patch_shape)
        pnum = 0
        for i in range(0, max_patch_num[0] + 1):
            for j in range(0, max_patch_num[1] + 1):
                for k in range(0, max_patch_num[2] + 1):
                    for p in range(0, max_patch_num[3] + 1):
                        slice_str = '[:, {}:{}, {}:{}, {}:{}, {}:{}]'.format(
                            i * patch_shape[0],
                            (i + 1) * patch_shape[0],
                            j * patch_shape[1],
                            (j + 1) * patch_shape[1],
                            k * patch_shape[2],
                            (k + 1) * patch_shape[2],
                            p * patch_shape[3],
                            (p + 1) * patch_shape[3],
                        )
                        with self.subTest(f"Patch: ({i},{j},{k},{p}) data{slice_str}"):
                            data_out_test = self.run_get_patch(pnum)
                            data_out_test = np.array(data_out_test).reshape(pshape)
                            data_out_true = self.data_out['data_out'][
                                :,
                                i * patch_shape[0] : (i + 1) * patch_shape[0],
                                j * patch_shape[1] : (j + 1) * patch_shape[1],
                                k * patch_shape[2] : (k + 1) * patch_shape[2],
                                p * patch_shape[3] : (p + 1) * patch_shape[3],
                            ]
                            self.assertTrue(
                                np.array_equal(data_out_test, data_out_true),
                            )
                        pnum += 1


if __name__ == '__main__':
    unittest.main()
