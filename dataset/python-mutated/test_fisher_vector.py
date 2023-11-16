import pytest
import numpy as np
pytest.importorskip('sklearn')
from skimage.feature._fisher_vector import learn_gmm, fisher_vector, FisherVectorException, DescriptorException

def test_gmm_wrong_descriptor_format_1():
    if False:
        while True:
            i = 10
    'Test that DescriptorException is raised when wrong type for descriptions\n    is passed.\n    '
    with pytest.raises(DescriptorException):
        learn_gmm('completely wrong test', n_modes=1)

def test_gmm_wrong_descriptor_format_2():
    if False:
        return 10
    'Test that DescriptorException is raised when descriptors are of\n    different dimensionality.\n    '
    with pytest.raises(DescriptorException):
        learn_gmm([np.zeros((5, 11)), np.zeros((4, 10))], n_modes=1)

def test_gmm_wrong_descriptor_format_3():
    if False:
        for i in range(10):
            print('nop')
    'Test that DescriptorException is raised when not all descriptors are of\n    rank 2.\n    '
    with pytest.raises(DescriptorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10, 1))], n_modes=1)

def test_gmm_wrong_descriptor_format_4():
    if False:
        i = 10
        return i + 15
    'Test that DescriptorException is raised when elements of descriptor list\n    are of the incorrect type (i.e. not a NumPy ndarray).\n    '
    with pytest.raises(DescriptorException):
        learn_gmm([[1, 2, 3], [1, 2, 3]], n_modes=1)

def test_gmm_wrong_num_modes_format_1():
    if False:
        return 10
    'Test that FisherVectorException is raised when incorrect type for\n    n_modes is passed into the learn_gmm function.\n    '
    with pytest.raises(FisherVectorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10))], n_modes='not_valid')

def test_gmm_wrong_num_modes_format_2():
    if False:
        print('Hello World!')
    'Test that FisherVectorException is raised when a number that is not a\n    positive integer is passed into the n_modes argument of learn_gmm.\n    '
    with pytest.raises(FisherVectorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10))], n_modes=-1)

def test_gmm_wrong_covariance_type():
    if False:
        i = 10
        return i + 15
    'Test that FisherVectorException is raised when wrong covariance type is\n    passed in as a keyword argument.\n    '
    with pytest.raises(FisherVectorException):
        learn_gmm(np.random.random((10, 10)), n_modes=2, gm_args={'covariance_type': 'full'})

def test_gmm_correct_covariance_type():
    if False:
        for i in range(10):
            print('nop')
    'Test that GMM estimation is successful when the correct covariance type\n    is passed in as a keyword argument.\n    '
    gmm = learn_gmm(np.random.random((10, 10)), n_modes=2, gm_args={'covariance_type': 'diag'})
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.weights_ is not None

def test_gmm_e2e():
    if False:
        while True:
            i = 10
    '\n    Test the GMM estimation. Since this is essentially a wrapper for the\n    scikit-learn GaussianMixture class, the testing of the actual inner\n    workings of the GMM estimation is left to scikit-learn and its\n    dependencies.\n\n    We instead simply assert that the estimation was successful based on the\n    fact that the GMM object will have associated mixture weights, means, and\n    variances after estimation is successful/complete.\n    '
    gmm = learn_gmm(np.random.random((100, 64)), n_modes=5)
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.weights_ is not None

def test_fv_wrong_descriptor_types():
    if False:
        while True:
            i = 10
    '\n    Test that DescriptorException is raised when the incorrect type for the\n    descriptors is passed into the fisher_vector function.\n    '
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print('scikit-learn is not installed. Please ensure it is installed in order to use the Fisher vector functionality.')
    with pytest.raises(DescriptorException):
        fisher_vector([[1, 2, 3, 4]], GaussianMixture())

def test_fv_wrong_gmm_type():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that FisherVectorException is raised when a GMM not of type\n    sklearn.mixture.GaussianMixture is passed into the fisher_vector\n    function.\n    '

    class MyDifferentGaussianMixture:
        pass
    with pytest.raises(FisherVectorException):
        fisher_vector(np.zeros((10, 10)), MyDifferentGaussianMixture())

def test_fv_e2e():
    if False:
        i = 10
        return i + 15
    '\n    Test the Fisher vector computation given a GMM returned from the learn_gmm\n    function. We simply assert that the dimensionality of the resulting Fisher\n    vector is correct.\n\n    The dimensionality of a Fisher vector is given by 2KD + K, where K is the\n    number of Gaussians specified in the associated GMM, and D is the\n    dimensionality of the descriptors using to estimate the GMM.\n    '
    dim = 128
    num_modes = 8
    expected_dim = 2 * num_modes * dim + num_modes
    descriptors = [np.random.random((np.random.randint(5, 30), dim)) for _ in range(10)]
    gmm = learn_gmm(descriptors, n_modes=num_modes)
    fisher_vec = fisher_vector(descriptors[0], gmm)
    assert len(fisher_vec) == expected_dim

def test_fv_e2e_improved():
    if False:
        i = 10
        return i + 15
    '\n    Test the improved Fisher vector computation given a GMM returned from the\n    learn_gmm function. We simply assert that the dimensionality of the\n    resulting Fisher vector is correct.\n\n    The dimensionality of a Fisher vector is given by 2KD + K, where K is the\n    number of Gaussians specified in the associated GMM, and D is the\n    dimensionality of the descriptors using to estimate the GMM.\n    '
    dim = 128
    num_modes = 8
    expected_dim = 2 * num_modes * dim + num_modes
    descriptors = [np.random.random((np.random.randint(5, 30), dim)) for _ in range(10)]
    gmm = learn_gmm(descriptors, n_modes=num_modes)
    fisher_vec = fisher_vector(descriptors[0], gmm, improved=True)
    assert len(fisher_vec) == expected_dim