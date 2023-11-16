"""
Pythran implementation of columns grouping for finite difference Jacobian
estimation. Used by ._numdiff.group_columns and based on the Cython version.
"""
import numpy as np

def group_dense(m, n, A):
    if False:
        while True:
            i = 10
    B = A.T
    groups = -np.ones(n, dtype=np.intp)
    current_group = 0
    union = np.empty(m, dtype=np.intp)
    for i in range(n):
        if groups[i] >= 0:
            continue
        groups[i] = current_group
        all_grouped = True
        union[:] = B[i]
        for j in range(groups.shape[0]):
            if groups[j] < 0:
                all_grouped = False
            else:
                continue
            intersect = False
            for k in range(m):
                if union[k] > 0 and B[j, k] > 0:
                    intersect = True
                    break
            if not intersect:
                union += B[j]
                groups[j] = current_group
        if all_grouped:
            break
        current_group += 1
    return groups

def group_sparse(m, n, indices, indptr):
    if False:
        while True:
            i = 10
    groups = -np.ones(n, dtype=np.intp)
    current_group = 0
    union = np.empty(m, dtype=np.intp)
    for i in range(n):
        if groups[i] >= 0:
            continue
        groups[i] = current_group
        all_grouped = True
        union.fill(0)
        for k in range(indptr[i], indptr[i + 1]):
            union[indices[k]] = 1
        for j in range(groups.shape[0]):
            if groups[j] < 0:
                all_grouped = False
            else:
                continue
            intersect = False
            for k in range(indptr[j], indptr[j + 1]):
                if union[indices[k]] == 1:
                    intersect = True
                    break
            if not intersect:
                for k in range(indptr[j], indptr[j + 1]):
                    union[indices[k]] = 1
                groups[j] = current_group
        if all_grouped:
            break
        current_group += 1
    return groups