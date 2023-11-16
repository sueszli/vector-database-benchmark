"""Forest covertype dataset.

A classic dataset for classification benchmarks, featuring categorical and
real-valued features.

The dataset page is available from UCI Machine Learning Repository

    https://archive.ics.uci.edu/ml/datasets/Covertype

Courtesy of Jock A. Blackard and Colorado State University.
"""
import logging
import os
from gzip import GzipFile
from os.path import exists, join
from tempfile import TemporaryDirectory
import joblib
import numpy as np
from ..utils import Bunch, check_random_state
from ..utils._param_validation import validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _convert_data_dataframe, _fetch_remote, _pkl_filepath, load_descr
ARCHIVE = RemoteFileMetadata(filename='covtype.data.gz', url='https://ndownloader.figshare.com/files/5976039', checksum='614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771')
logger = logging.getLogger(__name__)
FEATURE_NAMES = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
FEATURE_NAMES += [f'Wilderness_Area_{i}' for i in range(4)]
FEATURE_NAMES += [f'Soil_Type_{i}' for i in range(40)]
TARGET_NAMES = ['Cover_Type']

@validate_params({'data_home': [str, os.PathLike, None], 'download_if_missing': ['boolean'], 'random_state': ['random_state'], 'shuffle': ['boolean'], 'return_X_y': ['boolean'], 'as_frame': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_covtype(*, data_home=None, download_if_missing=True, random_state=None, shuffle=False, return_X_y=False, as_frame=False):
    if False:
        for i in range(10):
            print('nop')
    "Load the covertype dataset (classification).\n\n    Download it if necessary.\n\n    =================   ============\n    Classes                        7\n    Samples total             581012\n    Dimensionality                54\n    Features                     int\n    =================   ============\n\n    Read more in the :ref:`User Guide <covtype_dataset>`.\n\n    Parameters\n    ----------\n    data_home : str or path-like, default=None\n        Specify another download and cache folder for the datasets. By default\n        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.\n\n    download_if_missing : bool, default=True\n        If False, raise an OSError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    random_state : int, RandomState instance or None, default=None\n        Determines random number generation for dataset shuffling. Pass an int\n        for reproducible output across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    shuffle : bool, default=False\n        Whether to shuffle dataset.\n\n    return_X_y : bool, default=False\n        If True, returns ``(data.data, data.target)`` instead of a Bunch\n        object.\n\n        .. versionadded:: 0.20\n\n    as_frame : bool, default=False\n        If True, the data is a pandas DataFrame including columns with\n        appropriate dtypes (numeric). The target is a pandas DataFrame or\n        Series depending on the number of target columns. If `return_X_y` is\n        True, then (`data`, `target`) will be pandas DataFrames or Series as\n        described below.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    dataset : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        data : ndarray of shape (581012, 54)\n            Each row corresponds to the 54 features in the dataset.\n        target : ndarray of shape (581012,)\n            Each value corresponds to one of\n            the 7 forest covertypes with values\n            ranging between 1 to 7.\n        frame : dataframe of shape (581012, 55)\n            Only present when `as_frame=True`. Contains `data` and `target`.\n        DESCR : str\n            Description of the forest covertype dataset.\n        feature_names : list\n            The names of the dataset columns.\n        target_names: list\n            The names of the target columns.\n\n    (data, target) : tuple if ``return_X_y`` is True\n        A tuple of two ndarray. The first containing a 2D array of\n        shape (n_samples, n_features) with each row representing one\n        sample and each column representing the features. The second\n        ndarray of shape (n_samples,) containing the target samples.\n\n        .. versionadded:: 0.20\n    "
    data_home = get_data_home(data_home=data_home)
    covtype_dir = join(data_home, 'covertype')
    samples_path = _pkl_filepath(covtype_dir, 'samples')
    targets_path = _pkl_filepath(covtype_dir, 'targets')
    available = exists(samples_path) and exists(targets_path)
    if download_if_missing and (not available):
        os.makedirs(covtype_dir, exist_ok=True)
        with TemporaryDirectory(dir=covtype_dir) as temp_dir:
            logger.info(f'Downloading {ARCHIVE.url}')
            archive_path = _fetch_remote(ARCHIVE, dirname=temp_dir)
            Xy = np.genfromtxt(GzipFile(filename=archive_path), delimiter=',')
            X = Xy[:, :-1]
            y = Xy[:, -1].astype(np.int32, copy=False)
            samples_tmp_path = _pkl_filepath(temp_dir, 'samples')
            joblib.dump(X, samples_tmp_path, compress=9)
            os.rename(samples_tmp_path, samples_path)
            targets_tmp_path = _pkl_filepath(temp_dir, 'targets')
            joblib.dump(y, targets_tmp_path, compress=9)
            os.rename(targets_tmp_path, targets_path)
    elif not available and (not download_if_missing):
        raise OSError('Data not found and `download_if_missing` is False')
    try:
        (X, y)
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)
    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]
    fdescr = load_descr('covtype.rst')
    frame = None
    if as_frame:
        (frame, X, y) = _convert_data_dataframe(caller_name='fetch_covtype', data=X, target=y, feature_names=FEATURE_NAMES, target_names=TARGET_NAMES)
    if return_X_y:
        return (X, y)
    return Bunch(data=X, target=y, frame=frame, target_names=TARGET_NAMES, feature_names=FEATURE_NAMES, DESCR=fdescr)