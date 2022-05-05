'''Tests real data'''

import unittest
import numpy as np

from npy_patcher import PatcherFloat


def get_test_data(filepath):
    '''Testing: more complex (3D) shapes, differing qspace indexing

    Datatype: long
    Padding required: (0, 0, 4, 3, 2, 1)
    '''
    data_out = np.load(filepath)
    data_out = np.pad(data_out, ((0, 0), (3, 2), (3, 3), (3, 2)))
    data_out = np.stack(
        [data_out[50], data_out[4], data_out[8], data_out[200], data_out[103],
        data_out[30], data_out[230], data_out[1]]
    )
    data_dict = {
        'qdx': np.array([50, 4, 8, 200, 103, 30, 230, 1]),
        'patch_shape': (10, 10, 10),
        'max_patch_num': (14, 17, 14),
        'data_out': data_out,
        'padding': (0, 0, 3, 2, 3, 3, 3, 2),
    }

    return data_dict


class RealDataTest(unittest.TestCase):
    '''Real data test class'''

    def setUp(self) -> None:
        self.set_up_vars()
        self.data_dict = get_test_data(self.filepath)

    def set_up_vars(self):
        '''Sets up variables for testing'''
        self.filepath = '/mnt/Data/Work/HCP/100206/Diffusion/data.npy'
        self.patcher = PatcherFloat()

    def run_get_patch(self, pnum):
        '''Runs patcher call given patch number'''
        return self.patcher.get_patch(
            self.filepath,
            self.data_dict['qdx'],
            self.data_dict['patch_shape'],
            pnum
        )

    def test_equality_loop(self):
        '''Tests whether arrays are equal'''
        max_patch_num = self.data_dict['max_patch_num']
        patch_shape = self.data_dict['patch_shape']
        pshape = (len(self.data_dict['qdx']), ) + tuple(patch_shape)
        for i in range(0, max_patch_num[0]+ 1):
            for j in range(0, max_patch_num[1] + 1):
                for k in range(0, max_patch_num[2] + 1):
                    slice_str = '[:, {}:{}, {}:{}, {}:{}]'.format(
                        i*patch_shape[0],
                        (i+1)*patch_shape[0],
                        j*patch_shape[1],
                        (j+1)*patch_shape[1],
                        k*patch_shape[2],
                        (k+1)*patch_shape[2],
                    )
                    with self.subTest(f'Patch: ({i},{j},{k}) data{slice_str}'):
                        print(f'Patch: ({i},{j},{k}) data{slice_str}')
                        data_out_test = self.run_get_patch(pnum=(i, j, k))
                        data_out_test = np.array(data_out_test).reshape(pshape)
                        data_out_true = self.data_dict['data_out'][
                            :,
                            i*patch_shape[0]:(i+1)*patch_shape[0],
                            j*patch_shape[1]:(j+1)*patch_shape[1],
                            k*patch_shape[2]:(k+1)*patch_shape[2],
                        ]
                        self.assertTrue(
                            np.array_equal(data_out_test, data_out_true),
                        )


if __name__ == '__main__':
    unittest.main()
