'''Testing Patcher class'''
import os
import unittest
import numpy as np

from skimage.util import view_as_windows

from npy_patcher import PatcherInt, PatcherLong


def get_test_data_2d(filepath):
    '''Testing: 2D shape with non-contiguous qspace indexing

    Datatype: int
    Padding required: (1, 0, 1, 0)
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.int32)
    data_in = np.pad(data_in, ((0, 0), (1, 1), (1, 1)), constant_values=42)
    rand = lambda x: np.random.randint(0, 50, (x, 5, 5), dtype=np.int32)
    data_in = np.concatenate([data_in, rand(4), data_in, data_in, rand(2)], axis=0)
    pshape = [3, 3]
    qidx = np.array([0, 5, 6])
    np.save(filepath, data_in, allow_pickle=False)
    extra_padding = ((1, 3), (6, 4))
    extra_padding = ((2, 0), (1, 3))
    data_in_dict = {
        'fpath': filepath,
        'pshape': pshape,
        'pstride': [2, 2],
        'qidx': qidx,
        'padding': list(sum(extra_padding, ())),
    }
    data_out = np.pad(data_in, ((0, 0),) + extra_padding)
    data_out = view_as_windows(
        data_out,
        (len(data_in),) + tuple(pshape),
        (len(data_in),) + tuple(data_in_dict['pstride']),
    )
    data_out = data_out.flatten().reshape((-1, len(data_in)) + tuple(pshape))
    data_out = data_out[:, qidx, ...]
    data_out_dict = {
        'pnums': 4,
        'padding': tuple(sum(extra_padding, ())),
        'data_out': data_out,
    }

    return data_in_dict, data_out_dict


def get_test_data_3d(filepath):
    '''Testing: more complex (3D) shapes, differing qspace indexing

    Datatype: long
    Padding required: (0, 0, 4, 3, 2, 1)
    '''
    data_in = np.arange(12 * 33 * 22).reshape(1, 12, 33, 22).astype(np.int64)
    rand = lambda x: np.random.randint(0, 300, (x, 12, 33, 22), dtype=np.int64)
    data_in_final = np.concatenate([data_in, rand(1), data_in * 2, rand(3), data_in * 3], axis=0)
    np.save(filepath, data_in_final, allow_pickle=False)
    qidx = np.array([6, 0, 2])
    pshape = (3, 10, 5)
    data_in_dict = {
        'fpath': filepath,
        'qidx': qidx,
        'pshape': pshape,
        'pstride': (3, 5, 4),
        'padding': [],
    }
    data_out = np.pad(data_in_final, ((0, 0), (0, 0), (1, 1), (2, 1)))
    data_out = view_as_windows(
        data_out,
        (7,) + tuple(pshape),
        (7,) + tuple(data_in_dict['pstride']),
    )
    pnums = 4 * 6 * 6
    data_out = data_out.reshape((pnums, 7) + tuple(pshape))
    data_out = data_out[:, qidx, ...]
    data_out_dict = {
        'pnums': pnums,
        'data_out': data_out,
        'padding': (0, 0, 1, 1, 2, 1),
    }

    return data_in_dict, data_out_dict


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

        def test_equality_loop(self):
            '''Tests equality of array output for each patch'''
            pshape = (len(self.data_in['qidx']),) + tuple(self.data_in['pshape'])
            for pnum in range(self.data_out['pnums']):
                with self.subTest(f'Patch: {pnum}'):
                    data_out_test = self.run_get_patch(pnum)
                    data_out_test = np.array(data_out_test).reshape(pshape)
                    data_out_true = self.data_out['data_out'][pnum, ...]
                    self.assertTrue(
                        np.array_equal(data_out_test, data_out_true),
                        f'\n{data_out_test}\n\n-------\n{data_out_true}',
                    )
                    self.assertEqual(tuple(self.patcher.get_padding()), self.data_out['padding'])


class TestPatcherLoop2D(BaseTestCases.BaseTest):
    '''2D test case testing each patch'''

    def set_up_vars(self):
        self.filepath = 'test_data_loop_2D.npy'
        self.setup_func = get_test_data_2d
        self.patcher = PatcherInt()


class TestPatcherLoop3D(BaseTestCases.BaseTest):
    '''2D test case testing each patch'''

    def set_up_vars(self):
        self.filepath = 'test_data_loop_3D.npy'
        self.setup_func = get_test_data_3d
        self.patcher = PatcherLong()


if __name__ == '__main__':
    unittest.main()
