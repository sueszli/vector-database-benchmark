"""Modified Olivetti faces dataset.

The original database was available from (now defunct)

    https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

The version retrieved here comes in MATLAB format from the personal
web page of Sam Roweis:

    https://cs.nyu.edu/~roweis/
"""
from os import PathLike, makedirs, remove
from os.path import exists
import joblib
import numpy as np
from scipy.io import loadmat
from ..utils import Bunch, check_random_state
from ..utils._param_validation import validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath, load_descr
FACES = RemoteFileMetadata(filename='olivettifaces.mat', url='https://ndownloader.figshare.com/files/5976027', checksum='b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794')

@validate_params({'data_home': [str, PathLike, None], 'shuffle': ['boolean'], 'random_state': ['random_state'], 'download_if_missing': ['boolean'], 'return_X_y': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_olivetti_faces(*, data_home=None, shuffle=False, random_state=0, download_if_missing=True, return_X_y=False):
    if False:
        return 10
    "Load the Olivetti faces data-set from AT&T (classification).\n\n    Download it if necessary.\n\n    =================   =====================\n    Classes                                40\n    Samples total                         400\n    Dimensionality                       4096\n    Features            real, between 0 and 1\n    =================   =====================\n\n    Read more in the :ref:`User Guide <olivetti_faces_dataset>`.\n\n    Parameters\n    ----------\n    data_home : str or path-like, default=None\n        Specify another download and cache folder for the datasets. By default\n        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.\n\n    shuffle : bool, default=False\n        If True the order of the dataset is shuffled to avoid having\n        images of the same person grouped.\n\n    random_state : int, RandomState instance or None, default=0\n        Determines random number generation for dataset shuffling. Pass an int\n        for reproducible output across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    download_if_missing : bool, default=True\n        If False, raise an OSError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    return_X_y : bool, default=False\n        If True, returns `(data, target)` instead of a `Bunch` object. See\n        below for more information about the `data` and `target` object.\n\n        .. versionadded:: 0.22\n\n    Returns\n    -------\n    data : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        data: ndarray, shape (400, 4096)\n            Each row corresponds to a ravelled\n            face image of original size 64 x 64 pixels.\n        images : ndarray, shape (400, 64, 64)\n            Each row is a face image\n            corresponding to one of the 40 subjects of the dataset.\n        target : ndarray, shape (400,)\n            Labels associated to each face image.\n            Those labels are ranging from 0-39 and correspond to the\n            Subject IDs.\n        DESCR : str\n            Description of the modified Olivetti Faces Dataset.\n\n    (data, target) : tuple if `return_X_y=True`\n        Tuple with the `data` and `target` objects described above.\n\n        .. versionadded:: 0.22\n    "
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filepath = _pkl_filepath(data_home, 'olivetti.pkz')
    if not exists(filepath):
        if not download_if_missing:
            raise OSError('Data not found and `download_if_missing` is False')
        print('downloading Olivetti faces from %s to %s' % (FACES.url, data_home))
        mat_path = _fetch_remote(FACES, dirname=data_home)
        mfile = loadmat(file_name=mat_path)
        remove(mat_path)
        faces = mfile['faces'].T.copy()
        joblib.dump(faces, filepath, compress=6)
        del mfile
    else:
        faces = joblib.load(filepath)
    faces = np.float32(faces)
    faces = faces - faces.min()
    faces /= faces.max()
    faces = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    target = np.array([i // 10 for i in range(400)])
    if shuffle:
        random_state = check_random_state(random_state)
        order = random_state.permutation(len(faces))
        faces = faces[order]
        target = target[order]
    faces_vectorized = faces.reshape(len(faces), -1)
    fdescr = load_descr('olivetti_faces.rst')
    if return_X_y:
        return (faces_vectorized, target)
    return Bunch(data=faces_vectorized, images=faces, target=target, DESCR=fdescr)