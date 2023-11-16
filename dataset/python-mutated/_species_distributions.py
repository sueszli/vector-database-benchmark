"""
=============================
Species distribution dataset
=============================

This dataset represents the geographic distribution of species.
The dataset is provided by Phillips et. al. (2006).

The two species are:

 - `"Bradypus variegatus"
   <http://www.iucnredlist.org/details/3038/0>`_ ,
   the Brown-throated Sloth.

 - `"Microryzomys minutus"
   <http://www.iucnredlist.org/details/13408/0>`_ ,
   also known as the Forest Small Rice Rat, a rodent that lives in Peru,
   Colombia, Ecuador, Peru, and Venezuela.

References
----------

`"Maximum entropy modeling of species geographic distributions"
<http://rob.schapire.net/papers/ecolmod.pdf>`_ S. J. Phillips,
R. P. Anderson, R. E. Schapire - Ecological Modelling, 190:231-259, 2006.

Notes
-----

For an example of using this dataset, see
:ref:`examples/applications/plot_species_distribution_modeling.py
<sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.
"""
import logging
from io import BytesIO
from os import PathLike, makedirs, remove
from os.path import exists
import joblib
import numpy as np
from ..utils import Bunch
from ..utils._param_validation import validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath
SAMPLES = RemoteFileMetadata(filename='samples.zip', url='https://ndownloader.figshare.com/files/5976075', checksum='abb07ad284ac50d9e6d20f1c4211e0fd3c098f7f85955e89d321ee8efe37ac28')
COVERAGES = RemoteFileMetadata(filename='coverages.zip', url='https://ndownloader.figshare.com/files/5976078', checksum='4d862674d72e79d6cee77e63b98651ec7926043ba7d39dcb31329cf3f6073807')
DATA_ARCHIVE_NAME = 'species_coverage.pkz'
logger = logging.getLogger(__name__)

def _load_coverage(F, header_length=6, dtype=np.int16):
    if False:
        while True:
            i = 10
    'Load a coverage file from an open file object.\n\n    This will return a numpy array of the given dtype\n    '
    header = [F.readline() for _ in range(header_length)]
    make_tuple = lambda t: (t.split()[0], float(t.split()[1]))
    header = dict([make_tuple(line) for line in header])
    M = np.loadtxt(F, dtype=dtype)
    nodata = int(header[b'NODATA_value'])
    if nodata != -9999:
        M[nodata] = -9999
    return M

def _load_csv(F):
    if False:
        print('Hello World!')
    'Load csv file.\n\n    Parameters\n    ----------\n    F : file object\n        CSV file open in byte mode.\n\n    Returns\n    -------\n    rec : np.ndarray\n        record array representing the data\n    '
    names = F.readline().decode('ascii').strip().split(',')
    rec = np.loadtxt(F, skiprows=0, delimiter=',', dtype='a22,f4,f4')
    rec.dtype.names = names
    return rec

def construct_grids(batch):
    if False:
        for i in range(10):
            print('nop')
    'Construct the map grid from the batch object\n\n    Parameters\n    ----------\n    batch : Batch object\n        The object returned by :func:`fetch_species_distributions`\n\n    Returns\n    -------\n    (xgrid, ygrid) : 1-D arrays\n        The grid corresponding to the values in batch.coverages\n    '
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + batch.Nx * batch.grid_size
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + batch.Ny * batch.grid_size
    xgrid = np.arange(xmin, xmax, batch.grid_size)
    ygrid = np.arange(ymin, ymax, batch.grid_size)
    return (xgrid, ygrid)

@validate_params({'data_home': [str, PathLike, None], 'download_if_missing': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_species_distributions(*, data_home=None, download_if_missing=True):
    if False:
        for i in range(10):
            print('nop')
    'Loader for species distribution dataset from Phillips et. al. (2006).\n\n    Read more in the :ref:`User Guide <datasets>`.\n\n    Parameters\n    ----------\n    data_home : str or path-like, default=None\n        Specify another download and cache folder for the datasets. By default\n        all scikit-learn data is stored in \'~/scikit_learn_data\' subfolders.\n\n    download_if_missing : bool, default=True\n        If False, raise an OSError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    Returns\n    -------\n    data : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        coverages : array, shape = [14, 1592, 1212]\n            These represent the 14 features measured\n            at each point of the map grid.\n            The latitude/longitude values for the grid are discussed below.\n            Missing data is represented by the value -9999.\n        train : record array, shape = (1624,)\n            The training points for the data.  Each point has three fields:\n\n            - train[\'species\'] is the species name\n            - train[\'dd long\'] is the longitude, in degrees\n            - train[\'dd lat\'] is the latitude, in degrees\n        test : record array, shape = (620,)\n            The test points for the data.  Same format as the training data.\n        Nx, Ny : integers\n            The number of longitudes (x) and latitudes (y) in the grid\n        x_left_lower_corner, y_left_lower_corner : floats\n            The (x,y) position of the lower-left corner, in degrees\n        grid_size : float\n            The spacing between points of the grid, in degrees\n\n    Notes\n    -----\n\n    This dataset represents the geographic distribution of species.\n    The dataset is provided by Phillips et. al. (2006).\n\n    The two species are:\n\n    - `"Bradypus variegatus"\n      <http://www.iucnredlist.org/details/3038/0>`_ ,\n      the Brown-throated Sloth.\n\n    - `"Microryzomys minutus"\n      <http://www.iucnredlist.org/details/13408/0>`_ ,\n      also known as the Forest Small Rice Rat, a rodent that lives in Peru,\n      Colombia, Ecuador, Peru, and Venezuela.\n\n    - For an example of using this dataset with scikit-learn, see\n      :ref:`examples/applications/plot_species_distribution_modeling.py\n      <sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.\n\n    References\n    ----------\n\n    * `"Maximum entropy modeling of species geographic distributions"\n      <http://rob.schapire.net/papers/ecolmod.pdf>`_\n      S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,\n      190:231-259, 2006.\n    '
    data_home = get_data_home(data_home)
    if not exists(data_home):
        makedirs(data_home)
    extra_params = dict(x_left_lower_corner=-94.8, Nx=1212, y_left_lower_corner=-56.05, Ny=1592, grid_size=0.05)
    dtype = np.int16
    archive_path = _pkl_filepath(data_home, DATA_ARCHIVE_NAME)
    if not exists(archive_path):
        if not download_if_missing:
            raise OSError('Data not found and `download_if_missing` is False')
        logger.info('Downloading species data from %s to %s' % (SAMPLES.url, data_home))
        samples_path = _fetch_remote(SAMPLES, dirname=data_home)
        with np.load(samples_path) as X:
            for f in X.files:
                fhandle = BytesIO(X[f])
                if 'train' in f:
                    train = _load_csv(fhandle)
                if 'test' in f:
                    test = _load_csv(fhandle)
        remove(samples_path)
        logger.info('Downloading coverage data from %s to %s' % (COVERAGES.url, data_home))
        coverages_path = _fetch_remote(COVERAGES, dirname=data_home)
        with np.load(coverages_path) as X:
            coverages = []
            for f in X.files:
                fhandle = BytesIO(X[f])
                logger.debug(' - converting {}'.format(f))
                coverages.append(_load_coverage(fhandle))
            coverages = np.asarray(coverages, dtype=dtype)
        remove(coverages_path)
        bunch = Bunch(coverages=coverages, test=test, train=train, **extra_params)
        joblib.dump(bunch, archive_path, compress=9)
    else:
        bunch = joblib.load(archive_path)
    return bunch