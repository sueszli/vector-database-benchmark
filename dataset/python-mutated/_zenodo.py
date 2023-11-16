"""Collection of imbalanced datasets.

This collection of datasets has been proposed in [1]_. The
characteristics of the available datasets are presented in the table
below.

 ID    Name           Repository & Target           Ratio  #S       #F
 1     ecoli          UCI, target: imU              8.6:1  336      7
 2     optical_digits UCI, target: 8                9.1:1  5,620    64
 3     satimage       UCI, target: 4                9.3:1  6,435    36
 4     pen_digits     UCI, target: 5                9.4:1  10,992   16
 5     abalone        UCI, target: 7                9.7:1  4,177    10
 6     sick_euthyroid UCI, target: sick euthyroid   9.8:1  3,163    42
 7     spectrometer   UCI, target: >=44             11:1   531      93
 8     car_eval_34    UCI, target: good, v good     12:1   1,728    21
 9     isolet         UCI, target: A, B             12:1   7,797    617
 10    us_crime       UCI, target: >0.65            12:1   1,994    100
 11    yeast_ml8      LIBSVM, target: 8             13:1   2,417    103
 12    scene          LIBSVM, target: >one label    13:1   2,407    294
 13    libras_move    UCI, target: 1                14:1   360      90
 14    thyroid_sick   UCI, target: sick             15:1   3,772    52
 15    coil_2000      KDD, CoIL, target: minority   16:1   9,822    85
 16    arrhythmia     UCI, target: 06               17:1   452      278
 17    solar_flare_m0 UCI, target: M->0             19:1   1,389    32
 18    oil            UCI, target: minority         22:1   937      49
 19    car_eval_4     UCI, target: vgood            26:1   1,728    21
 20    wine_quality   UCI, wine, target: <=4        26:1   4,898    11
 21    letter_img     UCI, target: Z                26:1   20,000   16
 22    yeast_me2      UCI, target: ME2              28:1   1,484    8
 23    webpage        LIBSVM, w7a, target: minority 33:1   34,780   300
 24    ozone_level    UCI, ozone, data              34:1   2,536    72
 25    mammography    UCI, target: minority         42:1   11,183   6
 26    protein_homo   KDD CUP 2004, minority        111:1  145,751  74
 27    abalone_19     UCI, target: 19               130:1  4,177    10

References
----------
.. [1] Ding, Zejin, "Diversified Ensemble Classifiers for Highly
   Imbalanced Data Learning and their Application in Bioinformatics."
   Dissertation, Georgia State University, (2011).
"""
import tarfile
from collections import OrderedDict
from io import BytesIO
from os import makedirs
from os.path import isfile, join
from urllib.request import urlopen
import numpy as np
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch, check_random_state
from ..utils._param_validation import validate_params
URL = 'https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz'
PRE_FILENAME = 'x'
POST_FILENAME = 'data.npz'
MAP_NAME_ID_KEYS = ['ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'spectrometer', 'car_eval_34', 'isolet', 'us_crime', 'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick', 'coil_2000', 'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality', 'letter_img', 'yeast_me2', 'webpage', 'ozone_level', 'mammography', 'protein_homo', 'abalone_19']
MAP_NAME_ID = OrderedDict()
MAP_ID_NAME = OrderedDict()
for (v, k) in enumerate(MAP_NAME_ID_KEYS):
    MAP_NAME_ID[k] = v + 1
    MAP_ID_NAME[v + 1] = k

@validate_params({'data_home': [None, str], 'filter_data': [None, tuple], 'download_if_missing': ['boolean'], 'random_state': ['random_state'], 'shuffle': ['boolean'], 'verbose': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_datasets(*, data_home=None, filter_data=None, download_if_missing=True, random_state=None, shuffle=False, verbose=False):
    if False:
        while True:
            i = 10
    'Load the benchmark datasets from Zenodo, downloading it if necessary.\n\n    .. versionadded:: 0.3\n\n    Parameters\n    ----------\n    data_home : str, default=None\n        Specify another download and cache folder for the datasets. By default\n        all scikit-learn data is stored in \'~/scikit_learn_data\' subfolders.\n\n    filter_data : tuple of str/int, default=None\n        A tuple containing the ID or the name of the datasets to be returned.\n        Refer to the above table to get the ID and name of the datasets.\n\n    download_if_missing : bool, default=True\n        If False, raise a IOError if the data is not locally available\n        instead of trying to download the data from the source site.\n\n    random_state : int, RandomState instance or None, default=None\n        Random state for shuffling the dataset.\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by `np.random`.\n\n    shuffle : bool, default=False\n        Whether to shuffle dataset.\n\n    verbose : bool, default=False\n        Show information regarding the fetching.\n\n    Returns\n    -------\n    datasets : OrderedDict of Bunch object,\n        The ordered is defined by ``filter_data``. Each Bunch object ---\n        referred as dataset --- have the following attributes:\n\n        dataset.data : ndarray of shape (n_samples, n_features)\n\n        dataset.target : ndarray of shape (n_samples,)\n\n        dataset.DESCR : str\n            Description of the each dataset.\n\n    Notes\n    -----\n    This collection of datasets have been proposed in [1]_. The\n    characteristics of the available datasets are presented in the table\n    below.\n\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |ID|Name          | Repository & Target           | Ratio | #S      | #F  |\n    +==+==============+===============================+=======+=========+=====+\n    |1 |ecoli         | UCI, target: imU              | 8.6:1 | 336     | 7   |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |2 |optical_digits| UCI, target: 8                | 9.1:1 | 5,620   | 64  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |3 |satimage      | UCI, target: 4                | 9.3:1 | 6,435   | 36  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |4 |pen_digits    | UCI, target: 5                | 9.4:1 | 10,992  | 16  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |5 |abalone       | UCI, target: 7                | 9.7:1 | 4,177   | 10  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |6 |sick_euthyroid| UCI, target: sick euthyroid   | 9.8:1 | 3,163   | 42  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |7 |spectrometer  | UCI, target: >=44             | 11:1  | 531     | 93  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |8 |car_eval_34   | UCI, target: good, v good     | 12:1  | 1,728   | 21  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |9 |isolet        | UCI, target: A, B             | 12:1  | 7,797   | 617 |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |10|us_crime      | UCI, target: >0.65            | 12:1  | 1,994   | 100 |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |11|yeast_ml8     | LIBSVM, target: 8             | 13:1  | 2,417   | 103 |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |12|scene         | LIBSVM, target: >one label    | 13:1  | 2,407   | 294 |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |13|libras_move   | UCI, target: 1                | 14:1  | 360     | 90  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |14|thyroid_sick  | UCI, target: sick             | 15:1  | 3,772   | 52  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |15|coil_2000     | KDD, CoIL, target: minority   | 16:1  | 9,822   | 85  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |16|arrhythmia    | UCI, target: 06               | 17:1  | 452     | 278 |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |17|solar_flare_m0| UCI, target: M->0             | 19:1  | 1,389   | 32  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |18|oil           | UCI, target: minority         | 22:1  | 937     | 49  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |19|car_eval_4    | UCI, target: vgood            | 26:1  | 1,728   | 21  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |20|wine_quality  | UCI, wine, target: <=4        | 26:1  | 4,898   | 11  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |21|letter_img    | UCI, target: Z                | 26:1  | 20,000  | 16  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |22|yeast_me2     | UCI, target: ME2              | 28:1  | 1,484   | 8   |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |23|webpage       | LIBSVM, w7a, target: minority | 33:1  | 34,780  | 300 |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |24|ozone_level   | UCI, ozone, data              | 34:1  | 2,536   | 72  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |25|mammography   | UCI, target: minority         | 42:1  | 11,183  | 6   |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |26|protein_homo  | KDD CUP 2004, minority        | 111:1 | 145,751 | 74  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n    |27|abalone_19    | UCI, target: 19               | 130:1 | 4,177   | 10  |\n    +--+--------------+-------------------------------+-------+---------+-----+\n\n    References\n    ----------\n    .. [1] Ding, Zejin, "Diversified Ensemble Classifiers for Highly\n       Imbalanced Data Learning and their Application in Bioinformatics."\n       Dissertation, Georgia State University, (2011).\n    '
    data_home = get_data_home(data_home=data_home)
    zenodo_dir = join(data_home, 'zenodo')
    datasets = OrderedDict()
    if filter_data is None:
        filter_data_ = MAP_NAME_ID.keys()
    else:
        list_data = MAP_NAME_ID.keys()
        filter_data_ = []
        for it in filter_data:
            if isinstance(it, str):
                if it not in list_data:
                    raise ValueError(f'{it} is not a dataset available. The available datasets are {list_data}')
                else:
                    filter_data_.append(it)
            elif isinstance(it, int):
                if it < 1 or it > 27:
                    raise ValueError(f'The dataset with the ID={it} is not an available dataset. The IDs are {range(1, 28)}')
                else:
                    filter_data_.append(MAP_ID_NAME[it])
            else:
                raise ValueError(f'The value in the tuple should be str or int. Got {type(it)} instead.')
    for it in filter_data_:
        filename = PRE_FILENAME + str(MAP_NAME_ID[it]) + POST_FILENAME
        filename = join(zenodo_dir, filename)
        available = isfile(filename)
        if download_if_missing and (not available):
            makedirs(zenodo_dir, exist_ok=True)
            if verbose:
                print('Downloading %s' % URL)
            f = BytesIO(urlopen(URL).read())
            tar = tarfile.open(fileobj=f)
            tar.extractall(path=zenodo_dir)
        elif not download_if_missing and (not available):
            raise IOError('Data not found and `download_if_missing` is False')
        data = np.load(filename)
        (X, y) = (data['data'], data['label'])
        if shuffle:
            ind = np.arange(X.shape[0])
            rng = check_random_state(random_state)
            rng.shuffle(ind)
            X = X[ind]
            y = y[ind]
        datasets[it] = Bunch(data=X, target=y, DESCR=it)
    return datasets