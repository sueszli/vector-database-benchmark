"""Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/
"""
import logging
from numbers import Integral, Real
from os import PathLike, listdir, makedirs, remove
from os.path import exists, isdir, join
import numpy as np
from joblib import Memory
from ..utils import Bunch
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ._base import RemoteFileMetadata, _fetch_remote, get_data_home, load_descr
logger = logging.getLogger(__name__)
ARCHIVE = RemoteFileMetadata(filename='lfw.tgz', url='https://ndownloader.figshare.com/files/5976018', checksum='055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0')
FUNNELED_ARCHIVE = RemoteFileMetadata(filename='lfw-funneled.tgz', url='https://ndownloader.figshare.com/files/5976015', checksum='b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a')
TARGETS = (RemoteFileMetadata(filename='pairsDevTrain.txt', url='https://ndownloader.figshare.com/files/5976012', checksum='1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa'), RemoteFileMetadata(filename='pairsDevTest.txt', url='https://ndownloader.figshare.com/files/5976009', checksum='7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c'), RemoteFileMetadata(filename='pairs.txt', url='https://ndownloader.figshare.com/files/5976006', checksum='ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592'))

def _check_fetch_lfw(data_home=None, funneled=True, download_if_missing=True):
    if False:
        while True:
            i = 10
    'Helper function to download any missing LFW data'
    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, 'lfw_home')
    if not exists(lfw_home):
        makedirs(lfw_home)
    for target in TARGETS:
        target_filepath = join(lfw_home, target.filename)
        if not exists(target_filepath):
            if download_if_missing:
                logger.info('Downloading LFW metadata: %s', target.url)
                _fetch_remote(target, dirname=lfw_home)
            else:
                raise OSError('%s is missing' % target_filepath)
    if funneled:
        data_folder_path = join(lfw_home, 'lfw_funneled')
        archive = FUNNELED_ARCHIVE
    else:
        data_folder_path = join(lfw_home, 'lfw')
        archive = ARCHIVE
    if not exists(data_folder_path):
        archive_path = join(lfw_home, archive.filename)
        if not exists(archive_path):
            if download_if_missing:
                logger.info('Downloading LFW data (~200MB): %s', archive.url)
                _fetch_remote(archive, dirname=lfw_home)
            else:
                raise OSError('%s is missing' % archive_path)
        import tarfile
        logger.debug('Decompressing the data archive to %s', data_folder_path)
        tarfile.open(archive_path, 'r:gz').extractall(path=lfw_home)
        remove(archive_path)
    return (lfw_home, data_folder_path)

def _load_imgs(file_paths, slice_, color, resize):
    if False:
        for i in range(10):
            print('nop')
    'Internally used to load images'
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('The Python Imaging Library (PIL) is required to load data from jpeg files. Please refer to https://pillow.readthedocs.io/en/stable/installation.html for installing PIL.')
    default_slice = (slice(0, 250), slice(0, 250))
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple((s or ds for (s, ds) in zip(slice_, default_slice)))
    (h_slice, w_slice) = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)
    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)
    n_faces = len(file_paths)
    if not color:
        faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else:
        faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)
    for (i, file_path) in enumerate(file_paths):
        if i % 1000 == 0:
            logger.debug('Loading face #%05d / %05d', i + 1, n_faces)
        pil_img = Image.open(file_path)
        pil_img = pil_img.crop((w_slice.start, h_slice.start, w_slice.stop, h_slice.stop))
        if resize is not None:
            pil_img = pil_img.resize((w, h))
        face = np.asarray(pil_img, dtype=np.float32)
        if face.ndim == 0:
            raise RuntimeError('Failed to read the image file %s, Please make sure that libjpeg is installed' % file_path)
        face /= 255.0
        if not color:
            face = face.mean(axis=2)
        faces[i, ...] = face
    return faces

def _fetch_lfw_people(data_folder_path, slice_=None, color=False, resize=None, min_faces_per_person=0):
    if False:
        while True:
            i = 10
    'Perform the actual data loading for the lfw people dataset\n\n    This operation is meant to be cached by a joblib wrapper.\n    '
    (person_names, file_paths) = ([], [])
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)
    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError('min_faces_per_person=%d is too restrictive' % min_faces_per_person)
    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)
    faces = _load_imgs(file_paths, slice_, color, resize)
    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    (faces, target) = (faces[indices], target[indices])
    return (faces, target, target_names)

@validate_params({'data_home': [str, PathLike, None], 'funneled': ['boolean'], 'resize': [Interval(Real, 0, None, closed='neither'), None], 'min_faces_per_person': [Interval(Integral, 0, None, closed='left'), None], 'color': ['boolean'], 'slice_': [tuple, Hidden(None)], 'download_if_missing': ['boolean'], 'return_X_y': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_lfw_people(*, data_home=None, funneled=True, resize=0.5, min_faces_per_person=0, color=False, slice_=(slice(70, 195), slice(78, 172)), download_if_missing=True, return_X_y=False):
    if False:
        print('Hello World!')
    "Load the Labeled Faces in the Wild (LFW) people dataset (classification).\n\n    Download it if necessary.\n\n    =================   =======================\n    Classes                                5749\n    Samples total                         13233\n    Dimensionality                         5828\n    Features            real, between 0 and 255\n    =================   =======================\n\n    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.\n\n    Parameters\n    ----------\n    data_home : str or path-like, default=None\n        Specify another download and cache folder for the datasets. By default\n        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.\n\n    funneled : bool, default=True\n        Download and use the funneled variant of the dataset.\n\n    resize : float or None, default=0.5\n        Ratio used to resize the each face picture. If `None`, no resizing is\n        performed.\n\n    min_faces_per_person : int, default=None\n        The extracted dataset will only retain pictures of people that have at\n        least `min_faces_per_person` different pictures.\n\n    color : bool, default=False\n        Keep the 3 RGB channels instead of averaging them to a single\n        gray level channel. If color is True the shape of the data has\n        one more dimension than the shape with color = False.\n\n    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))\n        Provide a custom 2D slice (height, width) to extract the\n        'interesting' part of the jpeg files and avoid use statistical\n        correlation from the background.\n\n    download_if_missing : bool, default=True\n        If False, raise an OSError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    return_X_y : bool, default=False\n        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch\n        object. See below for more information about the `dataset.data` and\n        `dataset.target` object.\n\n        .. versionadded:: 0.20\n\n    Returns\n    -------\n    dataset : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        data : numpy array of shape (13233, 2914)\n            Each row corresponds to a ravelled face image\n            of original size 62 x 47 pixels.\n            Changing the ``slice_`` or resize parameters will change the\n            shape of the output.\n        images : numpy array of shape (13233, 62, 47)\n            Each row is a face image corresponding to one of the 5749 people in\n            the dataset. Changing the ``slice_``\n            or resize parameters will change the shape of the output.\n        target : numpy array of shape (13233,)\n            Labels associated to each face image.\n            Those labels range from 0-5748 and correspond to the person IDs.\n        target_names : numpy array of shape (5749,)\n            Names of all persons in the dataset.\n            Position in array corresponds to the person ID in the target array.\n        DESCR : str\n            Description of the Labeled Faces in the Wild (LFW) dataset.\n\n    (data, target) : tuple if ``return_X_y`` is True\n        A tuple of two ndarray. The first containing a 2D array of\n        shape (n_samples, n_features) with each row representing one\n        sample and each column representing the features. The second\n        ndarray of shape (n_samples,) containing the target samples.\n\n        .. versionadded:: 0.20\n    "
    (lfw_home, data_folder_path) = _check_fetch_lfw(data_home=data_home, funneled=funneled, download_if_missing=download_if_missing)
    logger.debug('Loading LFW people faces from %s', lfw_home)
    m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)
    (faces, target, target_names) = load_func(data_folder_path, resize=resize, min_faces_per_person=min_faces_per_person, color=color, slice_=slice_)
    X = faces.reshape(len(faces), -1)
    fdescr = load_descr('lfw.rst')
    if return_X_y:
        return (X, target)
    return Bunch(data=X, images=faces, target=target, target_names=target_names, DESCR=fdescr)

def _fetch_lfw_pairs(index_file_path, data_folder_path, slice_=None, color=False, resize=None):
    if False:
        while True:
            i = 10
    'Perform the actual data loading for the LFW pairs dataset\n\n    This operation is meant to be cached by a joblib wrapper.\n    '
    with open(index_file_path, 'rb') as index_file:
        split_lines = [ln.decode().strip().split('\t') for ln in index_file]
    pair_specs = [sl for sl in split_lines if len(sl) > 2]
    n_pairs = len(pair_specs)
    target = np.zeros(n_pairs, dtype=int)
    file_paths = list()
    for (i, components) in enumerate(pair_specs):
        if len(components) == 3:
            target[i] = 1
            pair = ((components[0], int(components[1]) - 1), (components[0], int(components[2]) - 1))
        elif len(components) == 4:
            target[i] = 0
            pair = ((components[0], int(components[1]) - 1), (components[2], int(components[3]) - 1))
        else:
            raise ValueError('invalid line %d: %r' % (i + 1, components))
        for (j, (name, idx)) in enumerate(pair):
            try:
                person_folder = join(data_folder_path, name)
            except TypeError:
                person_folder = join(data_folder_path, str(name, 'UTF-8'))
            filenames = list(sorted(listdir(person_folder)))
            file_path = join(person_folder, filenames[idx])
            file_paths.append(file_path)
    pairs = _load_imgs(file_paths, slice_, color, resize)
    shape = list(pairs.shape)
    n_faces = shape.pop(0)
    shape.insert(0, 2)
    shape.insert(0, n_faces // 2)
    pairs.shape = shape
    return (pairs, target, np.array(['Different persons', 'Same person']))

@validate_params({'subset': [StrOptions({'train', 'test', '10_folds'})], 'data_home': [str, PathLike, None], 'funneled': ['boolean'], 'resize': [Interval(Real, 0, None, closed='neither'), None], 'color': ['boolean'], 'slice_': [tuple, Hidden(None)], 'download_if_missing': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_lfw_pairs(*, subset='train', data_home=None, funneled=True, resize=0.5, color=False, slice_=(slice(70, 195), slice(78, 172)), download_if_missing=True):
    if False:
        print('Hello World!')
    'Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).\n\n    Download it if necessary.\n\n    =================   =======================\n    Classes                                   2\n    Samples total                         13233\n    Dimensionality                         5828\n    Features            real, between 0 and 255\n    =================   =======================\n\n    In the official `README.txt`_ this task is described as the\n    "Restricted" task.  As I am not sure as to implement the\n    "Unrestricted" variant correctly, I left it as unsupported for now.\n\n      .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt\n\n    The original images are 250 x 250 pixels, but the default slice and resize\n    arguments reduce them to 62 x 47.\n\n    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.\n\n    Parameters\n    ----------\n    subset : {\'train\', \'test\', \'10_folds\'}, default=\'train\'\n        Select the dataset to load: \'train\' for the development training\n        set, \'test\' for the development test set, and \'10_folds\' for the\n        official evaluation set that is meant to be used with a 10-folds\n        cross validation.\n\n    data_home : str or path-like, default=None\n        Specify another download and cache folder for the datasets. By\n        default all scikit-learn data is stored in \'~/scikit_learn_data\'\n        subfolders.\n\n    funneled : bool, default=True\n        Download and use the funneled variant of the dataset.\n\n    resize : float, default=0.5\n        Ratio used to resize the each face picture.\n\n    color : bool, default=False\n        Keep the 3 RGB channels instead of averaging them to a single\n        gray level channel. If color is True the shape of the data has\n        one more dimension than the shape with color = False.\n\n    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))\n        Provide a custom 2D slice (height, width) to extract the\n        \'interesting\' part of the jpeg files and avoid use statistical\n        correlation from the background.\n\n    download_if_missing : bool, default=True\n        If False, raise an OSError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    Returns\n    -------\n    data : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        data : ndarray of shape (2200, 5828). Shape depends on ``subset``.\n            Each row corresponds to 2 ravel\'d face images\n            of original size 62 x 47 pixels.\n            Changing the ``slice_``, ``resize`` or ``subset`` parameters\n            will change the shape of the output.\n        pairs : ndarray of shape (2200, 2, 62, 47). Shape depends on ``subset``\n            Each row has 2 face images corresponding\n            to same or different person from the dataset\n            containing 5749 people. Changing the ``slice_``,\n            ``resize`` or ``subset`` parameters will change the shape of the\n            output.\n        target : numpy array of shape (2200,). Shape depends on ``subset``.\n            Labels associated to each pair of images.\n            The two label values being different persons or the same person.\n        target_names : numpy array of shape (2,)\n            Explains the target values of the target array.\n            0 corresponds to "Different person", 1 corresponds to "same person".\n        DESCR : str\n            Description of the Labeled Faces in the Wild (LFW) dataset.\n    '
    (lfw_home, data_folder_path) = _check_fetch_lfw(data_home=data_home, funneled=funneled, download_if_missing=download_if_missing)
    logger.debug('Loading %s LFW pairs from %s', subset, lfw_home)
    m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_pairs)
    label_filenames = {'train': 'pairsDevTrain.txt', 'test': 'pairsDevTest.txt', '10_folds': 'pairs.txt'}
    if subset not in label_filenames:
        raise ValueError("subset='%s' is invalid: should be one of %r" % (subset, list(sorted(label_filenames.keys()))))
    index_file_path = join(lfw_home, label_filenames[subset])
    (pairs, target, target_names) = load_func(index_file_path, data_folder_path, resize=resize, color=color, slice_=slice_)
    fdescr = load_descr('lfw.rst')
    return Bunch(data=pairs.reshape(len(pairs), -1), pairs=pairs, target=target, target_names=target_names, DESCR=fdescr)