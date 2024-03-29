'''Tests real data'''

import unittest
import numpy as np

from npy_patcher import PatcherFloat


def generate_qspace_sample():
    '''Generates random qspace sample between size 6 & 30'''
    qdim_size = np.random.randint(6, 30, dtype=np.int32)
    return np.random.choice(288, qdim_size, replace=False)


def pretty_str_array(arr):
    '''Formats array for nice printing'''
    return '[{}, {}, ..., {}, {}]'.format(
        str(arr[0]).ljust(3, ' '),
        str(arr[1]).ljust(3, ' '),
        str(arr[-2]).ljust(3, ' '),
        str(arr[-1]).ljust(3, ' '),
    )


def pretty_str_index(first, last, maxlen=3, comma=True):
    if comma:
        return f'{first}:{last},'.ljust((maxlen * 2) + 2, ' ')
    return f'{first}:{last}'.ljust((maxlen * 2) + 1, ' ')


def get_test_data(filepath):
    '''Testing: more complex (3D) shapes, differing qspace indexing

    Datatype: long
    Padding required: (0, 0, 4, 3, 2, 1)
    '''
    data_out = np.load(filepath)
    data_out = np.pad(data_out, ((0, 0), (3, 2), (3, 3), (3, 2)))
    data_in = {
        'fpath': filepath,
        'pshape': (10, 10, 10),
        'pstride': (10, 10, 10),
        'padding': [],
    }
    data_out = {
        'max_patch_num': (14, 17, 14),
        'data_out': data_out,
        'padding': (0, 0, 3, 2, 3, 3, 3, 2),
    }

    return data_in, data_out


class RealDataTest(unittest.TestCase):
    '''Real data test class'''

    def setUp(self) -> None:
        self.set_up_vars()
        self.data_in, self.data_out = get_test_data(self.filepath)

    def set_up_vars(self):
        '''Sets up variables for testing'''
        self.filepath = '/mnt/Data/Work/HCP/100206/Diffusion/data.npy'
        self.patcher = PatcherFloat()

    def run_get_patch(self, pnum, qdx):
        '''Runs patcher call given patch number'''
        return self.patcher.get_patch(qidx=qdx, pnum=pnum, **self.data_in)

    def test_equality_loop(self):
        '''Tests whether arrays are equal'''
        max_patch_num = self.data_out['max_patch_num']
        patch_shape = self.data_in['pshape']
        pnum = 0
        for i in range(0, max_patch_num[0] + 1):
            for j in range(0, max_patch_num[1] + 1):
                for k in range(0, max_patch_num[2] + 1):
                    qdx = generate_qspace_sample()
                    pshape = (len(qdx),) + tuple(patch_shape)
                    slice_str = '[ {}, {} {} {}]'.format(
                        pretty_str_array(qdx),
                        pretty_str_index(i * patch_shape[0], (i + 1) * patch_shape[0]),
                        pretty_str_index(j * patch_shape[1], (j + 1) * patch_shape[1]),
                        pretty_str_index(k * patch_shape[2], (k + 1) * patch_shape[2], comma=False),
                    )
                    with self.subTest(f'Patch: ({i},{j},{k}) data{slice_str}'):
                        patch_str = f'({i},{j},{k})'.ljust(10, ' ')
                        print(f'Patch: {patch_str} data{slice_str}')
                        data_out_test = self.run_get_patch(pnum, qdx)
                        data_out_test = np.array(data_out_test).reshape(pshape)
                        data_out_true = self.data_out['data_out'][
                            :,
                            i * patch_shape[0] : (i + 1) * patch_shape[0],
                            j * patch_shape[1] : (j + 1) * patch_shape[1],
                            k * patch_shape[2] : (k + 1) * patch_shape[2],
                        ][qdx, ...]
                        self.assertTrue(
                            np.array_equal(data_out_test, data_out_true),
                        )
                    pnum += 1


if __name__ == '__main__':
    unittest.main()
