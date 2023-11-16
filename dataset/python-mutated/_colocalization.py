import numpy as np
from scipy.stats import pearsonr
from .._shared.utils import check_shape_equality, as_binary_ndarray
__all__ = ['pearson_corr_coeff', 'manders_coloc_coeff', 'manders_overlap_coeff', 'intersection_coeff']

def pearson_corr_coeff(image0, image1, mask=None):
    if False:
        print('Hello World!')
    "Calculate Pearson's Correlation Coefficient between pixel intensities\n    in channels.\n\n    Parameters\n    ----------\n    image0 : (M, N) ndarray\n        Image of channel A.\n    image1 : (M, N) ndarray\n        Image of channel 2 to be correlated with channel B.\n        Must have same dimensions as `image0`.\n    mask : (M, N) ndarray of dtype bool, optional\n        Only `image0` and `image1` pixels within this region of interest mask\n        are included in the calculation. Must have same dimensions as `image0`.\n\n    Returns\n    -------\n    pcc : float\n        Pearson's correlation coefficient of the pixel intensities between\n        the two images, within the mask if provided.\n    p-value : float\n        Two-tailed p-value.\n\n    Notes\n    -----\n    Pearson's Correlation Coefficient (PCC) measures the linear correlation\n    between the pixel intensities of the two images. Its value ranges from -1\n    for perfect linear anti-correlation to +1 for perfect linear correlation.\n    The calculation of the p-value assumes that the intensities of pixels in\n    each input image are normally distributed.\n\n    Scipy's implementation of Pearson's correlation coefficient is used. Please\n    refer to it for further information and caveats [1]_.\n\n    .. math::\n        r = \\frac{\\sum (A_i - m_A_i) (B_i - m_B_i)}\n        {\\sqrt{\\sum (A_i - m_A_i)^2 \\sum (B_i - m_B_i)^2}}\n\n    where\n        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`\n        :math:`B_i` is the value of the :math:`i^{th}` pixel in `image1`,\n        :math:`m_A_i` is the mean of the pixel values in `image0`\n        :math:`m_B_i` is the mean of the pixel values in `image1`\n\n    A low PCC value does not necessarily mean that there is no correlation\n    between the two channel intensities, just that there is no linear\n    correlation. You may wish to plot the pixel intensities of each of the two\n    channels in a 2D scatterplot and use Spearman's rank correlation if a\n    non-linear correlation is visually identified [2]_. Also consider if you\n    are interested in correlation or co-occurence, in which case a method\n    involving segmentation masks (e.g. MCC or intersection coefficient) may be\n    more suitable [3]_ [4]_.\n\n    Providing the mask of only relevant sections of the image (e.g., cells, or\n    particular cellular compartments) and removing noise is important as the\n    PCC is sensitive to these measures [3]_ [4]_.\n\n    References\n    ----------\n    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html  # noqa\n    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html  # noqa\n    .. [3] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical\n           guide to evaluating colocalization in biological microscopy.\n           American journal of physiology. Cell physiology, 300(4), C723–C742.\n           https://doi.org/10.1152/ajpcell.00462.2010\n    .. [4] Bolte, S. and Cordelières, F.P. (2006), A guided tour into\n           subcellular colocalization analysis in light microscopy. Journal of\n           Microscopy, 224: 213-232.\n           https://doi.org/10.1111/j.1365-2818.2006.01706.x\n    "
    image0 = np.asarray(image0)
    image1 = np.asarray(image1)
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name='mask')
        check_shape_equality(image0, image1, mask)
        image0 = image0[mask]
        image1 = image1[mask]
    else:
        check_shape_equality(image0, image1)
        image0 = image0.reshape(-1)
        image1 = image1.reshape(-1)
    return pearsonr(image0, image1)

def manders_coloc_coeff(image0, image1_mask, mask=None):
    if False:
        print('Hello World!')
    "Manders' colocalization coefficient between two channels.\n\n    Parameters\n    ----------\n    image0 : (M, N) ndarray\n        Image of channel A. All pixel values should be non-negative.\n    image1_mask : (M, N) ndarray of dtype bool\n        Binary mask with segmented regions of interest in channel B.\n        Must have same dimensions as `image0`.\n    mask : (M, N) ndarray of dtype bool, optional\n        Only `image0` pixel values within this region of interest mask are\n        included in the calculation.\n        Must have same dimensions as `image0`.\n\n    Returns\n    -------\n    mcc : float\n        Manders' colocalization coefficient.\n\n    Notes\n    -----\n    Manders' Colocalization Coefficient (MCC) is the fraction of total\n    intensity of a certain channel (channel A) that is within the segmented\n    region of a second channel (channel B) [1]_. It ranges from 0 for no\n    colocalisation to 1 for complete colocalization. It is also referred to\n    as M1 and M2.\n\n    MCC is commonly used to measure the colocalization of a particular protein\n    in a subceullar compartment. Typically a segmentation mask for channel B\n    is generated by setting a threshold that the pixel values must be above\n    to be included in the MCC calculation. In this implementation,\n    the channel B mask is provided as the argument `image1_mask`, allowing\n    the exact segmentation method to be decided by the user beforehand.\n\n    The implemented equation is:\n\n    .. math::\n        r = \\frac{\\sum A_{i,coloc}}{\\sum A_i}\n\n    where\n        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`\n        :math:`A_{i,coloc} = A_i` if :math:`Bmask_i > 0`\n        :math:`Bmask_i` is the value of the :math:`i^{th}` pixel in\n        `mask`\n\n    MCC is sensitive to noise, with diffuse signal in the first channel\n    inflating its value. Images should be processed to remove out of focus and\n    background light before the MCC is calculated [2]_.\n\n    References\n    ----------\n    .. [1] Manders, E.M.M., Verbeek, F.J. and Aten, J.A. (1993), Measurement of\n           co-localization of objects in dual-colour confocal images. Journal\n           of Microscopy, 169: 375-382.\n           https://doi.org/10.1111/j.1365-2818.1993.tb03313.x\n           https://imagej.net/media/manders.pdf\n    .. [2] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical\n           guide to evaluating colocalization in biological microscopy.\n           American journal of physiology. Cell physiology, 300(4), C723–C742.\n           https://doi.org/10.1152/ajpcell.00462.2010\n\n    "
    image0 = np.asarray(image0)
    image1_mask = as_binary_ndarray(image1_mask, variable_name='image1_mask')
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name='mask')
        check_shape_equality(image0, image1_mask, mask)
        image0 = image0[mask]
        image1_mask = image1_mask[mask]
    else:
        check_shape_equality(image0, image1_mask)
    if image0.min() < 0:
        raise ValueError('image contains negative values')
    sum = np.sum(image0)
    if sum == 0:
        return 0
    return np.sum(image0 * image1_mask) / sum

def manders_overlap_coeff(image0, image1, mask=None):
    if False:
        print('Hello World!')
    "Manders' overlap coefficient\n\n    Parameters\n    ----------\n    image0 : (M, N) ndarray\n        Image of channel A. All pixel values should be non-negative.\n    image1 : (M, N) ndarray\n        Image of channel B. All pixel values should be non-negative.\n        Must have same dimensions as `image0`\n    mask : (M, N) ndarray of dtype bool, optional\n        Only `image0` and `image1` pixel values within this region of interest\n        mask are included in the calculation.\n        Must have ♣same dimensions as `image0`.\n\n    Returns\n    -------\n    moc: float\n        Manders' Overlap Coefficient of pixel intensities between the two\n        images.\n\n    Notes\n    -----\n    Manders' Overlap Coefficient (MOC) is given by the equation [1]_:\n\n    .. math::\n        r = \\frac{\\sum A_i B_i}{\\sqrt{\\sum A_i^2 \\sum B_i^2}}\n\n    where\n        :math:`A_i` is the value of the :math:`i^{th}` pixel in `image0`\n        :math:`B_i` is the value of the :math:`i^{th}` pixel in `image1`\n\n    It ranges between 0 for no colocalization and 1 for complete colocalization\n    of all pixels.\n\n    MOC does not take into account pixel intensities, just the fraction of\n    pixels that have positive values for both channels[2]_ [3]_. Its usefulness\n    has been criticized as it changes in response to differences in both\n    co-occurence and correlation and so a particular MOC value could indicate\n    a wide range of colocalization patterns [4]_ [5]_.\n\n    References\n    ----------\n    .. [1] Manders, E.M.M., Verbeek, F.J. and Aten, J.A. (1993), Measurement of\n           co-localization of objects in dual-colour confocal images. Journal\n           of Microscopy, 169: 375-382.\n           https://doi.org/10.1111/j.1365-2818.1993.tb03313.x\n           https://imagej.net/media/manders.pdf\n    .. [2] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical\n           guide to evaluating colocalization in biological microscopy.\n           American journal of physiology. Cell physiology, 300(4), C723–C742.\n           https://doi.org/10.1152/ajpcell.00462.2010\n    .. [3] Bolte, S. and Cordelières, F.P. (2006), A guided tour into\n           subcellular colocalization analysis in light microscopy. Journal of\n           Microscopy, 224: 213-232.\n           https://doi.org/10.1111/j.1365-2818.2006.01\n    .. [4] Adler J, Parmryd I. (2010), Quantifying colocalization by\n           correlation: the Pearson correlation coefficient is\n           superior to the Mander's overlap coefficient. Cytometry A.\n           Aug;77(8):733-42.https://doi.org/10.1002/cyto.a.20896\n    .. [5] Adler, J, Parmryd, I. Quantifying colocalization: The case for\n           discarding the Manders overlap coefficient. Cytometry. 2021; 99:\n           910– 920. https://doi.org/10.1002/cyto.a.24336\n\n    "
    image0 = np.asarray(image0)
    image1 = np.asarray(image1)
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name='mask')
        check_shape_equality(image0, image1, mask)
        image0 = image0[mask]
        image1 = image1[mask]
    else:
        check_shape_equality(image0, image1)
    if image0.min() < 0:
        raise ValueError('image0 contains negative values')
    if image1.min() < 0:
        raise ValueError('image1 contains negative values')
    denom = (np.sum(np.square(image0)) * np.sum(np.square(image1))) ** 0.5
    return np.sum(np.multiply(image0, image1)) / denom

def intersection_coeff(image0_mask, image1_mask, mask=None):
    if False:
        i = 10
        return i + 15
    "Fraction of a channel's segmented binary mask that overlaps with a\n    second channel's segmented binary mask.\n\n    Parameters\n    ----------\n    image0_mask : (M, N) ndarray of dtype bool\n        Image mask of channel A.\n    image1_mask : (M, N) ndarray of dtype bool\n        Image mask of channel B.\n        Must have same dimensions as `image0_mask`.\n    mask : (M, N) ndarray of dtype bool, optional\n        Only `image0_mask` and `image1_mask` pixels within this region of\n        interest\n        mask are included in the calculation.\n        Must have same dimensions as `image0_mask`.\n\n    Returns\n    -------\n    Intersection coefficient, float\n        Fraction of `image0_mask` that overlaps with `image1_mask`.\n\n    "
    image0_mask = as_binary_ndarray(image0_mask, variable_name='image0_mask')
    image1_mask = as_binary_ndarray(image1_mask, variable_name='image1_mask')
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name='mask')
        check_shape_equality(image0_mask, image1_mask, mask)
        image0_mask = image0_mask[mask]
        image1_mask = image1_mask[mask]
    else:
        check_shape_equality(image0_mask, image1_mask)
    nonzero_image0 = np.count_nonzero(image0_mask)
    if nonzero_image0 == 0:
        return 0
    nonzero_joint = np.count_nonzero(np.logical_and(image0_mask, image1_mask))
    return nonzero_joint / nonzero_image0