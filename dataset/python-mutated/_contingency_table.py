import scipy.sparse as sparse
import numpy as np
__all__ = ['contingency_table']

def contingency_table(im_true, im_test, *, ignore_labels=None, normalize=False):
    if False:
        while True:
            i = 10
    '\n    Return the contingency table for all regions in matched segmentations.\n\n    Parameters\n    ----------\n    im_true : ndarray of int\n        Ground-truth label image, same shape as im_test.\n    im_test : ndarray of int\n        Test image.\n    ignore_labels : sequence of int, optional\n        Labels to ignore. Any part of the true image labeled with any of these\n        values will not be counted in the score.\n    normalize : bool\n        Determines if the contingency table is normalized by pixel count.\n\n    Returns\n    -------\n    cont : scipy.sparse.csr_matrix\n        A contingency table. `cont[i, j]` will equal the number of voxels\n        labeled `i` in `im_true` and `j` in `im_test`.\n    '
    if ignore_labels is None:
        ignore_labels = []
    im_test_r = im_test.reshape(-1)
    im_true_r = im_true.reshape(-1)
    data = np.isin(im_true_r, ignore_labels, invert=True).astype(float)
    if normalize:
        data /= np.count_nonzero(data)
    cont = sparse.coo_matrix((data, (im_true_r, im_test_r))).tocsr()
    return cont