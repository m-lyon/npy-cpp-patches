import numpy as np


def get_test_data_one(filepath):
    '''Testing: Simple test, shame input and output data
    
    Datatype: double
    Padding required: None
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.double)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.arange(9).reshape(1, 3, 3).astype(np.double)
    patch_shape = (3, 3)
    qspace_index = np.array([0])
    patch_num = (0, 0)

    return patch_shape, qspace_index, patch_num, data_out


def get_test_data_two(filepath):
    '''Testing: padding required, getting patch at one edge
    
    Datatype: float
    Padding required: (1, 1, 1, 1)
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.float32)
    data_in = np.pad(data_in, ((0, 0), (1, 1), (1, 1)), constant_values=42)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.array([[[0, 42, 3], [0, 42, 6], [0, 42, 42]]]).astype(np.float32)
    patch_shape = (3, 3)
    qspace_index = np.array([0])
    patch_num = (0, 1)

    return patch_shape, qspace_index, patch_num, data_out


def get_test_data_three(filepath):
    '''Testing: more complex qspace indexing
    
    Datatype: int
    Padding required: (1, 1, 1, 1)
    '''
    data_in = np.arange(9).reshape(1, 3, 3).astype(np.int32)
    data_in = np.pad(data_in, ((0, 0), (1, 1), (1, 1)), constant_values=42)
    rand = lambda x: np.random.randint(0, 50, (x, 5, 5))
    data_in = np.concatenate([data_in, rand(4), data_in, data_in, rand(2)], axis=0)
    np.save(filepath, data_in, allow_pickle=False)
    data_out = np.array([[[0, 42, 3], [0, 42, 6], [0, 42, 42]]]).astype(np.int32)
    data_out = np.concatenate([data_out, data_out, data_out], axis=0)
    patch_shape = (3, 3)
    qspace_index = np.array([0, 5, 6])
    patch_num = (0, 1)
    
    return patch_shape, qspace_index, patch_num, data_out


def get_test_data_four(filepath):
    '''Testing: more complex (3D) shapes, differing qspace indexing
    
    Datatype: long
    Padding required: (0, 0, 3, 3, 2, 1)
    '''
    data_in = np.arange(12*33*22).reshape(1, 12, 33, 22).astype(np.int64)
    rand = lambda x: np.random.randint(0, 300, (x, 12, 33, 22), dtype=np.int64)
    data_in_final = np.concatenate([data_in, rand(1), data_in*2, rand(3), data_in*3], axis=0)
    np.save(filepath, data_in_final, allow_pickle=False)
    data_out = data_in[:, 6:9, 27:33, 0:3]
    data_out = np.pad(data_out, ((0, 0), (0, 0), (0, 3), (2, 0)), constant_values=0)
    data_out = np.concatenate([data_out*3, data_out, data_out*2], axis=0)
    patch_shape = (3, 10, 5)
    patch_num = (2, 3, 0)
    qspace_index = np.array([6, 0, 2])

    return patch_shape, qspace_index, patch_num, data_out


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
    patch_shape = (6, 10, 5, 3)
    patch_num = (0, 0, 2, 0)
    qspace_index = np.array([5, 0])

    return patch_shape, qspace_index, patch_num, data_out
