import copy

import numpy as np
from numba import jit


def matrix_minor(arr, i, j):
    return np.delete(np.delete(arr, i, axis=0), j, axis=1)


def vector_minor(arr, i):
    return np.delete(arr, i, axis=0)


def check_symmetric(a, rtol=1e-11, atol=1e-11):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def princomp(A):
    """ performs principal components analysis
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables.

     Returns :
      coeff :
        is a p-by-p matrix, each column containing coefficients
        for one principal component.
      score :
        the principal component scores; that is, the representation
        of A in the principal component space. Rows of SCORE
        correspond to observations, columns to components.
      latent :
        a vector containing the eigenvalues
        of the covariance matrix of A.
     """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A.T, axis=1)).T  # subtract the mean (along columns)
    [latent, coeff] = np.linalg.eig(np.cov(M))  # attention:not always sorted
    score = np.dot(coeff.T, M)  # projection of the data in the new space
    return coeff, score, latent


def softmax(vector):
    exp_vector = np.exp(vector)
    return exp_vector / np.sum(exp_vector)


def softmaxM1(vector):
    exp_vector = np.exp(vector) - 1
    return exp_vector / np.sum(exp_vector)


def normalised_log(vector):
    log_vector = np.log(vector)
    return log_vector / np.sum(log_vector)


def normalised_log_0(vector):
    log_vector = np.log(vector + np.e)
    return log_vector / np.sum(log_vector)


def positive_variable_update(a, b, dt, method="reciprocal_saturation"):
    """
    update a according to b.

    If method is"proportional": when b>0, a converge to b with a speed (b-a) * dt; when b<0,     a converge to 0 with a
    speed a*b*dt.

    If method is "reciprocal_saturation": when b>0, a increase with a speed b/(1+a)*dt; when b<0, a converge to 0 with a
     speed a*b/(1+a)*dt.

    :param method: The method used for the updating
    :param a: array to update
    :param b: array to control update
    :param dt: speed of converge
    :return: updated a
    """
    if method == "proportional":
        b = np.array(b)
        b1 = copy.deepcopy(b)
        # b1 is the converge speed factor
        b1[b1 > 0] = 1
        b1[b1 < 0] = -b1[b1 < 0]
        b2 = copy.deepcopy(b)
        # b2 is the converge point
        b2[b2 < 0] = 0
        a += (b2 - a) * dt * b1
    if method == "reciprocal_saturation":
        a_orig = np.array(a)
        b = np.array(b)
        index_b_postive = b > 0
        index_b_negative = b < 0
        a[index_b_postive] += b[index_b_postive] / (1 + a_orig[index_b_postive]) * dt
        # a[index_b_negative] += a_orig[index_b_negative]*b[index_b_negative] / (1 + a_orig[index_b_negative]) * dt
        a[index_b_negative] = a_orig[index_b_negative] * np.exp(
            b[index_b_negative] / (1 + a_orig[index_b_negative]) * dt)
    assert not np.any(np.isnan(a)), "a contains nan"
    assert not np.any(np.isinf(a)), "a contains inf"
    assert not np.any(np.isneginf(a)), "a contains neginf"
    assert not np.any(a < 0), "a contains negative value"
    return a


# def rational_variable_update(a, b, dt, method="proportional"):
#     """
#     update a according to b.
#
#     If method is"proportional": a converge to b with speed (b-a) * dt.
#
#
#     :param method: The method used for the updating
#     :param a: array to update
#     :param b: array to control update
#     :param dt: speed of converge
#     :return: updated a
#     """
#     if method == "proportional":
#         b = np.array(b)
#         a += (b - a) * dt
#     return a
# @jit(nopython=True)
def rational_variable_update(a, b, dt, method="proportional"):
    """
    update a according to b.

    If method is"proportional": a converge to b with speed (b-a) * dt.


    :param method: The method used for the updating
    :param a: array to update
    :param b: array to control update
    :param dt: speed of converge
    :return: updated a
    """
    if method == "proportional":
        if len(a.shape) == 0:
            a += (b - a) * dt
        else:
            if len(a.shape) == 1:
                rational_variable_update_numba_1(a, b, dt)
            elif len(a.shape) == 2:
                rational_variable_update_numba_2(a, b, dt)
            elif len(a.shape) == 3 and isinstance(dt, float):
                rational_variable_update_numba_3(a, b, dt)
            elif len(a.shape) == 3 and len(dt.shape)==1:
                rational_variable_update_numba_4(a, b, dt)
            else:
                a += (b - a) * dt
    return a


@jit(nopython=True)
def rational_variable_update_numba_1(a, b, dt):
    for i in range(len(a)):
        a[i] += (b[i] - a[i]) * dt
    return a

@jit(nopython=True)
def rational_variable_update_numba_2(a, b, dt):
    for i in range(len(a)):
        print('i', i)
        for j in range(len(a[i])):
            a[i][j] += (b[i][j] - a[i][j]) * dt
    return a

@jit(nopython=True)
def rational_variable_update_numba_3(a, b, dt):
    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(len(a[i][j])):
                    a[i][j][k] += (b[i][j][k] - a[i][j][k]) * dt
    return a

@jit(nopython=True)
def rational_variable_update_numba_4(a, b, dt):
    for k, dt_k in enumerate(dt):
        if dt_k != 0:
            for i in range(len(a)):
                for j in range(len(a[i])):
                        a[i][j][k] += (b[i][j][k] - a[i][j][k]) * dt_k
    return a


def relu(x):
    return np.maximum(x, 0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# # @jit(nopython=True)
# def top_n(values, n):
#     top_values = values[0:n]
#     top_indexes = list(np.argsort(top_values)[::-1])
#     top_values = list(top_values[top_indexes])
#     for count_1, value in enumerate(values, start=n):
#         if value
#         for count_2, top_value in enumerate(top_values):
#             if value > top_value:
#                 top_values.insert(count_2, value)
#                 top_indexes.insert(count_2, count_1)
#                 if len(top_values) > n:
#                     top_values.pop()
#                     top_indexes.pop()
#             print(count_1, value, count_2, top_value, top_values, top_indexes)
#             if value > top_value:
#                 break
#     return top_values, top_indexes


if __name__ == "__main__":
    # arr = np.random.normal(0, 1, (4, 4))
    #
    # # tests
    # print('arr', arr)
    #
    # print('arr1', matrix_minor(arr, 0, 0))
    #
    # print('arr2', matrix_minor(arr, 0, 1))
    #
    # import matplotlib.pyplot as plt
    #
    # A = np.array([[2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],
    #               [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]])
    #
    # coeff, score, latent = princomp(A.T)
    #
    # plt.figure()
    # plt.subplot(121)
    # # every eigenvector describe the direction
    # # of a principal component.
    # m = np.mean(A, axis=1)
    # plt.plot([0, -coeff[0, 0] * 2] + m[0], [0, -coeff[0, 1] * 2] + m[1], '--k')
    # plt.plot([0, coeff[1, 0] * 2] + m[0], [0, coeff[1, 1] * 2] + m[1], '--k')
    # plt.plot(A[0, :], A[1, :], 'ob')  # the data
    # plt.axis('equal')
    # plt.subplot(122)
    # # new data
    # plt.plot(score[0, :], score[1, :], '*g')
    # plt.axis('equal')
    # plt.show(block=False)

    testa = np.arange(12)
    np.random.shuffle(testa)
    print(testa)
    testb = top_n(testa, 5)
    print(testb)
    pass
