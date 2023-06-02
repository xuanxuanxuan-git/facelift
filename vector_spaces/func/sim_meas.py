import numpy as np


def delta(a, alpha):
    if alpha <= 0:
        return 0

    a_len = np.linalg.norm(a, 2)
    if 0 < alpha < a_len:
        return alpha / a_len
    else:
        return 1


def part(A, k):
    A_len = 0
    m = A.shape[1]
    for i in range(0, m):
        A_len += np.linalg.norm(A[:, i], 2)
    partition = np.zeros((A.shape[0], k))
    for j in range(0, k):
        temp_A = ((j + 1) * A_len) / k
        for i in range(0, m):
            partition[:, j] += delta(A[:, i], temp_A) * A[:, i]
            temp_A -= np.linalg.norm(A[:, i], 2)

    return partition


def S(A, B, w, eps):
    sim = 0
    k_star = A.shape[1]
    branched = False
    for i in range(A.shape[1]):
        sim += w[i] * np.linalg.norm(A[:, i] - B[:, i], ord=2)
        temp = B - A[:, i][:, None]
        temp = np.sqrt(temp[0, :]**2 + temp[1, :]**2)
        dist = np.min(temp)
        if dist > eps and branched is False:
            k_star = i
            branched = True
    return sim, k_star
