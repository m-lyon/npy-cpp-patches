'''Testing Patcher class'''

import unittest
import numpy as np

from skimage.util import view_as_windows

from npy_patcher import PatcherFloat


def generate_qspace_sample():
    '''Generates random qspace sample between size 6 & 30'''
    qdim_size = np.random.randint(6, 30, dtype=np.int32)
    return np.random.choice(288, qdim_size, replace=False)


def get_test_data(filepath):
    '''Testing: more complex (3D) shapes, differing qspace indexing

    Datatype: long
    Padding required: (0, 0, 4, 3, 2, 1)
    '''
    data_out = np.load(filepath)
    data_out = np.pad(data_out, ((0, 0), (3, 2), (1, 0), (3, 2)))
    pshape = (20, 15, 10)
    data_in_dict = {
        'fpath': filepath,
        'pshape': pshape,
        'pstride': (10, 10, 10),
    }
    pnums = 14 * 17 * 15
    data_out = view_as_windows(
        data_out,
        (288,) + tuple(pshape),
        (288,) + tuple(data_in_dict['pstride']),
    )
    data_out = data_out.reshape((pnums, 288) + tuple(pshape))
    data_out_dict = {
        'pnums': pnums,
        'data_out': data_out,
        'padding': (3, 2, 1, 0, 3, 2),
    }

    return data_in_dict, data_out_dict


class RealDataTest(unittest.TestCase):
    '''Actual Base test class'''

    # pylint: disable=no-member

    def setUp(self) -> None:
        self.set_up_vars()
        self.data_in, self.data_out = get_test_data(self.filepath)

    def set_up_vars(self):
        '''Sets up variables for testing'''
        self.filepath = '/mnt/Data/Work/HCP/100206/Diffusion/data.npy'
        self.patcher = PatcherFloat()

    def run_get_patch(self, pnum, qdx):
        '''Runs patcher get patch given runtime params'''
        return self.patcher.get_patch(pnum=pnum, qidx=qdx, **self.data_in)

    def test_equality_loop(self):
        '''Tests equality of array output for each patch'''
        for pnum in range(self.data_out['pnums']):
            with self.subTest(f'Patch: {pnum}'):
                print(f'Patch: {pnum}')
                qdx = generate_qspace_sample()
                pshape = (len(qdx),) + tuple(self.data_in['pshape'])
                data_out_test = self.run_get_patch(pnum, qdx)
                data_out_test = np.array(data_out_test).reshape(pshape)
                data_out_true = self.data_out['data_out'][pnum, qdx, ...]
                self.assertTrue(np.array_equal(data_out_test, data_out_true))
                self.assertEqual(tuple(self.patcher.get_padding()), self.data_out['padding'])


if __name__ == '__main__':
    unittest.main()
