from .._shared.utils import check_shape_equality
from ._contingency_table import contingency_table
__all__ = ['adapted_rand_error']

def adapted_rand_error(image_true=None, image_test=None, *, table=None, ignore_labels=(0,), alpha=0.5):
    if False:
        while True:
            i = 10
    'Compute Adapted Rand error as defined by the SNEMI3D contest. [1]_\n\n    Parameters\n    ----------\n    image_true : ndarray of int\n        Ground-truth label image, same shape as im_test.\n    image_test : ndarray of int\n        Test image.\n    table : scipy.sparse array in crs format, optional\n        A contingency table built with skimage.evaluate.contingency_table.\n        If None, it will be computed on the fly.\n    ignore_labels : sequence of int, optional\n        Labels to ignore. Any part of the true image labeled with any of these\n        values will not be counted in the score.\n    alpha : float, optional\n        Relative weight given to precision and recall in the adapted Rand error\n        calculation.\n\n    Returns\n    -------\n    are : float\n        The adapted Rand error.\n    prec : float\n        The adapted Rand precision: this is the number of pairs of pixels that\n        have the same label in the test label image *and* in the true image,\n        divided by the number in the test image.\n    rec : float\n        The adapted Rand recall: this is the number of pairs of pixels that\n        have the same label in the test label image *and* in the true image,\n        divided by the number in the true image.\n\n    Notes\n    -----\n    Pixels with label 0 in the true segmentation are ignored in the score.\n\n    The adapted Rand error is calculated as follows:\n\n    :math:`1 - \\frac{\\sum_{ij} p_{ij}^{2}}{\\alpha \\sum_{k} s_{k}^{2} +\n    (1-\\alpha)\\sum_{k} t_{k}^{2}}`,\n    where :math:`p_{ij}` is the probability that a pixel has the same label\n    in the test image *and* in the true image, :math:`t_{k}` is the\n    probability that a pixel has label :math:`k` in the true image,\n    and :math:`s_{k}` is the probability that a pixel has label :math:`k`\n    in the test image.\n\n    Default behavior is to weight precision and recall equally in the\n    adapted Rand error calculation.\n    When alpha = 0, adapted Rand error = recall.\n    When alpha = 1, adapted Rand error = precision.\n\n\n    References\n    ----------\n    .. [1] Arganda-Carreras I, Turaga SC, Berger DR, et al. (2015)\n           Crowdsourcing the creation of image segmentation algorithms\n           for connectomics. Front. Neuroanat. 9:142.\n           :DOI:`10.3389/fnana.2015.00142`\n    '
    if image_test is not None and image_true is not None:
        check_shape_equality(image_true, image_test)
    if table is None:
        p_ij = contingency_table(image_true, image_test, ignore_labels=ignore_labels, normalize=False)
    else:
        p_ij = table
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be between 0 and 1')
    sum_p_ij2 = p_ij.data @ p_ij.data - p_ij.sum()
    a_i = p_ij.sum(axis=1).A.ravel()
    b_i = p_ij.sum(axis=0).A.ravel()
    sum_a2 = a_i @ a_i - a_i.sum()
    sum_b2 = b_i @ b_i - b_i.sum()
    precision = sum_p_ij2 / sum_a2
    recall = sum_p_ij2 / sum_b2
    fscore = sum_p_ij2 / (alpha * sum_a2 + (1 - alpha) * sum_b2)
    are = 1.0 - fscore
    return (are, precision, recall)