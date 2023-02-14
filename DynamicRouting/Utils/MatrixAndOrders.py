import numpy as np


def expand_vector(vector, extra_dim):
    expanded_vector = np.zeros(np.array(vector.shape) + extra_dim)
    expanded_vector[:-extra_dim] = vector
    return expanded_vector


def expand_matrix(matrix, extra_dim):
    expanded_matrix = np.zeros(np.array(matrix.shape) + extra_dim)
    expanded_matrix[:-extra_dim, :-extra_dim] = matrix
    return expanded_matrix


def bisect_left(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def matrix2DXmatrix3D(A, B, axis_to_keep_in_B=3):
    if axis_to_keep_in_B == 3:
        return np.matmul(A.ravel(order='F'), B.reshape((B.shape[0] * B.shape[1], B.shape[2]), order='F'))
    else:
        raise NotImplementedError

def matrix2D_mul_matrix3D(A, B, axis_to_keep_in_B=3):
    if axis_to_keep_in_B == 3:
        result = np.multiply(np.expand_dims(A.ravel(order='F'), axis=1),
                           B.reshape((B.shape[0] * B.shape[1], B.shape[2]), order='F'))
        return result.reshape((B.shape[0], B.shape[1], B.shape[2]), order='F')
    else:
        raise NotImplementedError

if __name__ == "__main__":
    dim = [2, 3, 4]
    A = np.arange(dim[0]*dim[1]).reshape((dim[0], dim[1]))
    B = np.arange(dim[0]*dim[1]*dim[2]).reshape((dim[0], dim[1], dim[2]))
    print('A \n', A)
    print('B \n', B)
    prod_AB = matrix2DXmatrix3D(A, B)
    print('prod_AB \n', prod_AB)
    mul_AB = matrix2D_mul_matrix3D(A, B)
    print('mul_AB \n', mul_AB)
    sum_mul_AB = mul_AB.sum(axis=0).sum(axis=0)
    print('sum_mul_AB \n', sum_mul_AB)
    pass
