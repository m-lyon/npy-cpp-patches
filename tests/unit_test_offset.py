'''Testing Patcher class'''
import os
import unittest
import numpy as np

from npy_patcher import PatcherFloat


def get_test_data(filepath):
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
        'qidx': (5, 0),
        'pnum': 0,
    }
    data_out = {
        'data_out': data_out_final,
        'padding': (1, 1, 2, 1, 0, 0, 1, 0),
        'data_strides': (4 * 5 * 20 * 7 * 4, 4 * 5 * 20 * 7, 4 * 5 * 20, 4 * 5, 4),
        'patch_strides': (10 * 5 * 3 * 4, 5 * 3 * 4, 3 * 4, 4),
    }

    return data_in, data_out


class BaseTestCases:
    '''Base test case class with TestClass members'''

    class BaseTest(unittest.TestCase):
        '''Actual Base test class'''

        assert_type = tuple

        # pylint: disable=no-member

        def setUp(self) -> None:
            self.filepath = 'test_data_five.npy'
            self.patcher = PatcherFloat()
            data_in, self.data_out = self.setup_func(self.filepath)
            self.data_in = {}
            for key, val in data_in.items():
                if isinstance(val, tuple):
                    self.data_in[key] = self.assert_type(val)
                else:
                    self.data_in[key] = val

        def tearDown(self):
            os.remove(self.filepath)

        @property
        def setup_func(self):
            '''Setup func'''
            raise NotImplementedError

        def run_get_patch(self):
            '''Runs patcher get patch given runtime params'''
            return self.patcher.get_patch(**self.data_in)

        def debug_vars(self):
            '''Runs debug subroutine'''
            self.patcher.debug_vars(**self.data_in)


class TestMixin:
    '''Finegrained test mixin class, use with BaseTest'''

    def test_equality(self):
        '''Tests equality of array output'''
        data_out_test = self.run_get_patch()
        patch_shape = (len(self.data_in['qidx']),) + tuple(self.data_in['pshape'])
        data_out_test = np.array(data_out_test).reshape(patch_shape)
        self.assertTrue(np.array_equal(data_out_test, self.data_out['data_out']))

    def test_padding(self):
        '''Tests padding value is correct'''
        self.debug_vars()
        padding = tuple(self.patcher.get_padding())
        self.assertEqual(padding, self.data_out['padding'])

    def test_data_strides(self):
        '''Tests strides vector output'''
        self.debug_vars()
        data_strides = tuple(self.patcher.get_data_strides())
        self.assertEqual(data_strides, self.data_out['data_strides'])

    def test_patch_strides(self):
        '''Tests patch strides vector output'''
        self.debug_vars()
        patch_strides = tuple(self.patcher.get_patch_strides())
        self.assertEqual(patch_strides, self.data_out['patch_strides'])


class TestPatcherOne(BaseTestCases.BaseTest, TestMixin):
    '''Patch offset 1 in smallest dim'''

    @property
    def setup_func(self):
        '''Setup func'''

        def get_test_data_one(filepath):
            '''Testing: Patch shape larger than data shape

            Datatype: float
            Padding required: (1, 1, 2, 1, 0, 0, 1, 0)
            '''
            data_in, data_out = get_test_data(filepath)
            data_in['pnum_offset'] = (0, 0, 0, 1)
            data_out['data_out'] = data_out['data_out'][:, 0:6, 0:10, 0:5, 3:6]

            return data_in, data_out

        return get_test_data_one


class TestPatcherTwo(BaseTestCases.BaseTest, TestMixin):
    '''Patch offset 1 in smallest dim w/ pnum = 1'''

    @property
    def setup_func(self):
        '''Setup func'''

        def get_test_data_two(filepath):
            '''Testing: Patch shape larger than data shape

            Datatype: float
            Padding required: (1, 1, 2, 1, 0, 0, 1, 0)
            '''
            data_in, data_out = get_test_data(filepath)
            data_in['pnum_offset'] = (0, 0, 0, 1)
            data_in['pnum'] = 1
            data_out['data_out'] = data_out['data_out'][:, 0:6, 0:10, 5:10, 0:3]

            return data_in, data_out

        return get_test_data_two


class TestPatcherThree(BaseTestCases.BaseTest, TestMixin):
    '''Patch offset 1 in second smallest dim'''

    @property
    def setup_func(self):
        '''Setup func'''

        def get_test_data_three(filepath):
            '''Testing: Patch shape larger than data shape

            Datatype: float
            Padding required: (1, 1, 2, 1, 0, 0, 1, 0)
            '''
            data_in, data_out = get_test_data(filepath)
            data_in['pnum_offset'] = (0, 0, 1, 0)
            data_out['data_out'] = data_out['data_out'][:, 0:6, 0:10, 5:10, 0:3]

            return data_in, data_out

        return get_test_data_three


class TestPatcherFour(BaseTestCases.BaseTest):
    '''Patch invalid offset 1 in third smallest dim'''

    @property
    def setup_func(self):
        '''Setup func'''

        def get_test_data_three(filepath):
            '''Testing: Patch shape larger than data shape

            Datatype: float
            Padding required: (1, 1, 2, 1, 0, 0, 1, 0)
            '''
            data_in, data_out = get_test_data(filepath)
            data_in['pnum_offset'] = (0, 1, 0, 1)
            data_out['data_out'] = data_out['data_out'][:, 0:6, 0:10, 5:10, 0:3]

            return data_in, data_out

        return get_test_data_three

    def test_raises_error(self):
        '''raises error test for invalid offset'''
        self.assertRaises(RuntimeError, self.run_get_patch)


class TestPatcherFive(BaseTestCases.BaseTest, TestMixin):
    '''Patch offset'''

    @property
    def setup_func(self):
        '''Setup func'''

        def get_test_data_three(filepath):
            '''Testing: Patch shape larger than data shape

            Datatype: float
            Padding required: (1, 1, 2, 1, 0, 0, 1, 0)
            '''
            data_in, data_out = get_test_data(filepath)
            data_in['pnum_offset'] = (0, 0, 2, 1)
            data_out['data_out'] = data_out['data_out'][:, 0:6, 0:10, 10:15, 3:6]

            return data_in, data_out

        return get_test_data_three


if __name__ == '__main__':
    unittest.main()
