"""Standard test images.

For more images, see

 - http://sipi.usc.edu/database/database.php

"""
import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
_LEGACY_DATA_DIR = osp.dirname(__file__)
_DISTRIBUTION_DIR = osp.dirname(_LEGACY_DATA_DIR)
try:
    from pooch import file_hash
except ModuleNotFoundError:

    def file_hash(fname, alg='sha256'):
        if False:
            i = 10
            return i + 15
        '\n        Calculate the hash of a given file.\n        Useful for checking if a file has changed or been corrupted.\n        Parameters\n        ----------\n        fname : str\n            The name of the file.\n        alg : str\n            The type of the hashing algorithm\n        Returns\n        -------\n        hash : str\n            The hash of the file.\n        Examples\n        --------\n        >>> fname = "test-file-for-hash.txt"\n        >>> with open(fname, "w") as f:\n        ...     __ = f.write("content of the file")\n        >>> print(file_hash(fname))\n        0fc74468e6a9a829f103d069aeb2bb4f8646bad58bf146bb0e3379b759ec4a00\n        >>> import os\n        >>> os.remove(fname)\n        '
        import hashlib
        if alg not in hashlib.algorithms_available:
            raise ValueError(f"Algorithm '{alg}' not available in hashlib")
        chunksize = 65536
        hasher = hashlib.new(alg)
        with open(fname, 'rb') as fin:
            buff = fin.read(chunksize)
            while buff:
                hasher.update(buff)
                buff = fin.read(chunksize)
        return hasher.hexdigest()

def _has_hash(path, expected_hash):
    if False:
        print('Hello World!')
    'Check if the provided path has the expected hash.'
    if not osp.exists(path):
        return False
    return file_hash(path) == expected_hash

def _create_image_fetcher():
    if False:
        print('Hello World!')
    try:
        import pooch
        if not hasattr(pooch, '__version__'):
            retry = {}
        else:
            retry = {'retry_if_failed': 3}
    except ImportError:
        return (None, _LEGACY_DATA_DIR)
    if '+git' in __version__:
        skimage_version_for_pooch = __version__.replace('.dev0+git', '+git')
    else:
        skimage_version_for_pooch = __version__.replace('.dev', '+')
    if '+' in skimage_version_for_pooch:
        url = 'https://github.com/scikit-image/scikit-image/raw/{version}/skimage/'
    else:
        url = 'https://github.com/scikit-image/scikit-image/raw/v{version}/skimage/'
    image_fetcher = pooch.create(path=pooch.os_cache('scikit-image'), base_url=url, version=skimage_version_for_pooch, version_dev='main', env='SKIMAGE_DATADIR', registry=registry, urls=registry_urls, **retry)
    data_dir = osp.join(str(image_fetcher.abspath), 'data')
    return (image_fetcher, data_dir)
(_image_fetcher, data_dir) = _create_image_fetcher()

def _skip_pytest_case_requiring_pooch(data_filename):
    if False:
        for i in range(10):
            print('nop')
    'If a test case is calling pooch, skip it.\n\n    This running the test suite in environments without internet\n    access, skipping only the tests that try to fetch external data.\n    '
    if 'PYTEST_CURRENT_TEST' in os.environ:
        import pytest
        pytest.skip(f'Unable to download {data_filename}', allow_module_level=True)

def _ensure_cache_dir(*, target_dir):
    if False:
        while True:
            i = 10
    "Prepare local cache directory if it doesn't exist already.\n\n    Creates::\n\n        /path/to/target_dir/\n                 └─ data/\n                    └─ README.txt\n    "
    os.makedirs(osp.join(target_dir, 'data'), exist_ok=True)
    readme_src = osp.join(_DISTRIBUTION_DIR, 'data/README.txt')
    readme_dest = osp.join(target_dir, 'data/README.txt')
    if not osp.exists(readme_dest):
        shutil.copy2(readme_src, readme_dest)

def _fetch(data_filename):
    if False:
        for i in range(10):
            print('nop')
    "Fetch a given data file from either the local cache or the repository.\n\n    This function provides the path location of the data file given\n    its name in the scikit-image repository. If a data file is not included in the\n    distribution and pooch is available, it is downloaded and cached.\n\n    Parameters\n    ----------\n    data_filename : str\n        Name of the file in the scikit-image repository. e.g.\n        'restoration/tess/camera_rl.npy'.\n\n    Returns\n    -------\n    file_path : str\n        Path of the local file.\n\n    Raises\n    ------\n    KeyError:\n        If the filename is not known to the scikit-image distribution.\n\n    ModuleNotFoundError:\n        If the filename is known to the scikit-image distribution but pooch\n        is not installed.\n\n    ConnectionError:\n        If scikit-image is unable to connect to the internet but the\n        dataset has not been downloaded yet.\n    "
    expected_hash = registry[data_filename]
    if _image_fetcher is None:
        cache_dir = osp.dirname(data_dir)
    else:
        cache_dir = str(_image_fetcher.abspath)
    cached_file_path = osp.join(cache_dir, data_filename)
    if _has_hash(cached_file_path, expected_hash):
        return cached_file_path
    legacy_file_path = osp.join(_DISTRIBUTION_DIR, data_filename)
    if _has_hash(legacy_file_path, expected_hash):
        return legacy_file_path
    if _image_fetcher is None:
        _skip_pytest_case_requiring_pooch(data_filename)
        raise ModuleNotFoundError('The requested file is part of the scikit-image distribution, but requires the installation of an optional dependency, pooch. To install pooch, use your preferred python package manager. Follow installation instruction found at https://scikit-image.org/docs/stable/user_guide/install.html')
    _ensure_cache_dir(target_dir=cache_dir)
    try:
        cached_file_path = _image_fetcher.fetch(data_filename)
        return cached_file_path
    except ConnectionError as err:
        _skip_pytest_case_requiring_pooch(data_filename)
        raise ConnectionError('Tried to download a scikit-image dataset, but no internet connection is available. To avoid this message in the future, try `skimage.data.download_all()` when you are connected to the internet.') from err

def download_all(directory=None):
    if False:
        for i in range(10):
            print('nop')
    'Download all datasets for use with scikit-image offline.\n\n    Scikit-image datasets are no longer shipped with the library by default.\n    This allows us to use higher quality datasets, while keeping the\n    library download size small.\n\n    This function requires the installation of an optional dependency, pooch,\n    to download the full dataset. Follow installation instruction found at\n\n        https://scikit-image.org/docs/stable/user_guide/install.html\n\n    Call this function to download all sample images making them available\n    offline on your machine.\n\n    Parameters\n    ----------\n    directory: path-like, optional\n        The directory where the dataset should be stored.\n\n    Raises\n    ------\n    ModuleNotFoundError:\n        If pooch is not install, this error will be raised.\n\n    Notes\n    -----\n    scikit-image will only search for images stored in the default directory.\n    Only specify the directory if you wish to download the images to your own\n    folder for a particular reason. You can access the location of the default\n    data directory by inspecting the variable ``skimage.data.data_dir``.\n    '
    if _image_fetcher is None:
        raise ModuleNotFoundError('To download all package data, scikit-image needs an optional dependency, pooch.To install pooch, follow our installation instructions found at https://scikit-image.org/docs/stable/user_guide/install.html')
    old_dir = _image_fetcher.path
    try:
        if directory is not None:
            directory = osp.expanduser(directory)
            _image_fetcher.path = directory
        _ensure_cache_dir(target_dir=_image_fetcher.path)
        for data_filename in _image_fetcher.registry:
            file_path = _fetch(data_filename)
            if not file_path.startswith(str(_image_fetcher.path)):
                dest_path = osp.join(_image_fetcher.path, data_filename)
                os.makedirs(osp.dirname(dest_path), exist_ok=True)
                shutil.copy2(file_path, dest_path)
    finally:
        _image_fetcher.path = old_dir

def lbp_frontal_face_cascade_filename():
    if False:
        for i in range(10):
            print('nop')
    'Return the path to the XML file containing the weak classifier cascade.\n\n    These classifiers were trained using LBP features. The file is part\n    of the OpenCV repository [1]_.\n\n    References\n    ----------\n    .. [1] OpenCV lbpcascade trained files\n           https://github.com/opencv/opencv/tree/master/data/lbpcascades\n    '
    return _fetch('data/lbpcascade_frontalface_opencv.xml')

def _load(f, as_gray=False):
    if False:
        for i in range(10):
            print('nop')
    'Load an image file located in the data directory.\n\n    Parameters\n    ----------\n    f : string\n        File name.\n    as_gray : bool, optional\n        Whether to convert the image to grayscale.\n\n    Returns\n    -------\n    img : ndarray\n        Image loaded from ``skimage.data_dir``.\n    '
    from ..io import imread
    return imread(_fetch(f), as_gray=as_gray)

def camera():
    if False:
        i = 10
        return i + 15
    'Gray-level "camera" image.\n\n    Can be used for segmentation and denoising examples.\n\n    Returns\n    -------\n    camera : (512, 512) uint8 ndarray\n        Camera image.\n\n    Notes\n    -----\n    No copyright restrictions. CC0 by the photographer (Lav Varshney).\n\n    .. versionchanged:: 0.18\n        This image was replaced due to copyright restrictions. For more\n        information, please see [1]_.\n\n    References\n    ----------\n    .. [1] https://github.com/scikit-image/scikit-image/issues/3927\n    '
    return _load('data/camera.png')

def eagle():
    if False:
        return 10
    'A golden eagle.\n\n    Suitable for examples on segmentation, Hough transforms, and corner\n    detection.\n\n    Notes\n    -----\n    No copyright restrictions. CC0 by the photographer (Dayane Machado).\n\n    Returns\n    -------\n    eagle : (2019, 1826) uint8 ndarray\n        Eagle image.\n    '
    return _load('data/eagle.png')

def astronaut():
    if False:
        while True:
            i = 10
    'Color image of the astronaut Eileen Collins.\n\n    Photograph of Eileen Collins, an American astronaut. She was selected\n    as an astronaut in 1992 and first piloted the space shuttle STS-63 in\n    1995. She retired in 2006 after spending a total of 38 days, 8 hours\n    and 10 minutes in outer space.\n\n    This image was downloaded from the NASA Great Images database\n    <https://flic.kr/p/r9qvLn>`__.\n\n    No known copyright restrictions, released into the public domain.\n\n    Returns\n    -------\n    astronaut : (512, 512, 3) uint8 ndarray\n        Astronaut image.\n    '
    return _load('data/astronaut.png')

def brick():
    if False:
        print('Hello World!')
    'Brick wall.\n\n    Returns\n    -------\n    brick : (512, 512) uint8 image\n        A small section of a brick wall.\n\n    Notes\n    -----\n    The original image was downloaded from\n    `CC0Textures <https://cc0textures.com/view.php?tex=Bricks25>`_ and licensed\n    under the Creative Commons CC0 License.\n\n    A perspective transform was then applied to the image, prior to\n    rotating it by 90 degrees, cropping and scaling it to obtain the final\n    image.\n    '
    "\n    The following code was used to obtain the final image.\n\n    >>> import sys; print(sys.version)\n    >>> import platform; print(platform.platform())\n    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')\n    >>> import numpy; print(f'numpy version: {numpy.__version__}')\n    >>> import imageio; print(f'imageio version {imageio.__version__}')\n    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)\n    [GCC 7.3.0]\n    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid\n    scikit-image version: 0.16.dev0\n    numpy version: 1.16.4\n    imageio version 2.4.1\n\n    >>> import requests\n    >>> import zipfile\n    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Bricks25/%5B2K%5DBricks25.zip'\n    >>> r = requests.get(url)\n    >>> with open('[2K]Bricks25.zip', 'bw') as f:\n    ...     f.write(r.content)\n    >>> with zipfile.ZipFile('[2K]Bricks25.zip') as z:\n    ... z.extract('Bricks25_col.jpg')\n\n    >>> from numpy.linalg import inv\n    >>> from skimage.transform import rescale, warp, rotate\n    >>> from skimage.color import rgb2gray\n    >>> from imageio import imread, imwrite\n    >>> from skimage import img_as_ubyte\n    >>> import numpy as np\n\n\n    >>> # Obtained playing around with GIMP 2.10 with their perspective tool\n    >>> H = inv(np.asarray([[ 0.54764, -0.00219, 0],\n    ...                     [-0.12822,  0.54688, 0],\n    ...                     [-0.00022,        0, 1]]))\n\n\n    >>> brick_orig = imread('Bricks25_col.jpg')\n    >>> brick = warp(brick_orig, H)\n    >>> brick = rescale(brick[:1024, :1024], (0.5, 0.5, 1))\n    >>> brick = rotate(brick, -90)\n    >>> imwrite('brick.png', img_as_ubyte(rgb2gray(brick)))\n    "
    return _load('data/brick.png', as_gray=True)

def grass():
    if False:
        while True:
            i = 10
    'Grass.\n\n    Returns\n    -------\n    grass : (512, 512) uint8 image\n        Some grass.\n\n    Notes\n    -----\n    The original image was downloaded from\n    `DeviantArt <https://www.deviantart.com/linolafett/art/Grass-01-434853879>`__\n    and licensed under the Creative Commons CC0 License.\n\n    The downloaded image was cropped to include a region of ``(512, 512)``\n    pixels around the top left corner, converted to grayscale, then to uint8\n    prior to saving the result in PNG format.\n\n    '
    "\n    The following code was used to obtain the final image.\n\n    >>> import sys; print(sys.version)\n    >>> import platform; print(platform.platform())\n    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')\n    >>> import numpy; print(f'numpy version: {numpy.__version__}')\n    >>> import imageio; print(f'imageio version {imageio.__version__}')\n    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)\n    [GCC 7.3.0]\n    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid\n    scikit-image version: 0.16.dev0\n    numpy version: 1.16.4\n    imageio version 2.4.1\n\n    >>> import requests\n    >>> import zipfile\n    >>> url = 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/a407467e-4ff0-49f1-923f-c9e388e84612/d76wfef-2878b78d-5dce-43f9-be36-26ec9bc0df3b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2E0MDc0NjdlLTRmZjAtNDlmMS05MjNmLWM5ZTM4OGU4NDYxMlwvZDc2d2ZlZi0yODc4Yjc4ZC01ZGNlLTQzZjktYmUzNi0yNmVjOWJjMGRmM2IuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.98hIcOTCqXWQ67Ec5bM5eovKEn2p91mWB3uedH61ynI'\n    >>> r = requests.get(url)\n    >>> with open('grass_orig.jpg', 'bw') as f:\n    ...     f.write(r.content)\n    >>> grass_orig = imageio.imread('grass_orig.jpg')\n    >>> grass = skimage.img_as_ubyte(skimage.color.rgb2gray(grass_orig[:512, :512]))\n    >>> imageio.imwrite('grass.png', grass)\n    "
    return _load('data/grass.png', as_gray=True)

def gravel():
    if False:
        i = 10
        return i + 15
    'Gravel\n\n    Returns\n    -------\n    gravel : (512, 512) uint8 image\n        Grayscale gravel sample.\n\n    Notes\n    -----\n    The original image was downloaded from\n    `CC0Textures <https://cc0textures.com/view.php?tex=Gravel04>`__ and\n    licensed under the Creative Commons CC0 License.\n\n    The downloaded image was then rescaled to ``(1024, 1024)``, then the\n    top left ``(512, 512)`` pixel region  was cropped prior to converting the\n    image to grayscale and uint8 data type. The result was saved using the\n    PNG format.\n    '
    "\n    The following code was used to obtain the final image.\n\n    >>> import sys; print(sys.version)\n    >>> import platform; print(platform.platform())\n    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')\n    >>> import numpy; print(f'numpy version: {numpy.__version__}')\n    >>> import imageio; print(f'imageio version {imageio.__version__}')\n    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)\n    [GCC 7.3.0]\n    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid\n    scikit-image version: 0.16.dev0\n    numpy version: 1.16.4\n    imageio version 2.4.1\n\n    >>> import requests\n    >>> import zipfile\n\n    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Gravel04/%5B2K%5DGravel04.zip'\n    >>> r = requests.get(url)\n    >>> with open('[2K]Gravel04.zip', 'bw') as f:\n    ...     f.write(r.content)\n\n    >>> with zipfile.ZipFile('[2K]Gravel04.zip') as z:\n    ...     z.extract('Gravel04_col.jpg')\n\n    >>> from skimage.transform import resize\n    >>> gravel_orig = imageio.imread('Gravel04_col.jpg')\n    >>> gravel = resize(gravel_orig, (1024, 1024))\n    >>> gravel = skimage.img_as_ubyte(skimage.color.rgb2gray(gravel[:512, :512]))\n    >>> imageio.imwrite('gravel.png', gravel)\n    "
    return _load('data/gravel.png', as_gray=True)

def text():
    if False:
        for i in range(10):
            print('nop')
    'Gray-level "text" image used for corner detection.\n\n    Notes\n    -----\n    This image was downloaded from Wikipedia\n    <https://en.wikipedia.org/wiki/File:Corner.png>`__.\n\n    No known copyright restrictions, released into the public domain.\n\n    Returns\n    -------\n    text : (172, 448) uint8 ndarray\n        Text image.\n    '
    return _load('data/text.png')

def checkerboard():
    if False:
        print('Hello World!')
    'Checkerboard image.\n\n    Checkerboards are often used in image calibration, since the\n    corner-points are easy to locate.  Because of the many parallel\n    edges, they also visualise distortions particularly well.\n\n    Returns\n    -------\n    checkerboard : (200, 200) uint8 ndarray\n        Checkerboard image.\n    '
    return _load('data/chessboard_GRAY.png')

def cells3d():
    if False:
        i = 10
        return i + 15
    '3D fluorescence microscopy image of cells.\n\n    The returned data is a 3D multichannel array with dimensions provided in\n    ``(z, c, y, x)`` order. Each voxel has a size of ``(0.29 0.26 0.26)``\n    micrometer. Channel 0 contains cell membranes, channel 1 contains nuclei.\n\n    Returns\n    -------\n    cells3d: (60, 2, 256, 256) uint16 ndarray\n        The volumetric images of cells taken with an optical microscope.\n\n    Notes\n    -----\n    The data for this was provided by the Allen Institute for Cell Science.\n\n    It has been downsampled by a factor of 4 in the row and column dimensions\n    to reduce computational time.\n\n    The microscope reports the following voxel spacing in microns:\n\n        * Original voxel size is ``(0.290, 0.065, 0.065)``.\n        * Scaling factor is ``(1, 4, 4)`` in each dimension.\n        * After rescaling the voxel size is ``(0.29 0.26 0.26)``.\n    '
    return _load('data/cells3d.tif')

def human_mitosis():
    if False:
        while True:
            i = 10
    'Image of human cells undergoing mitosis.\n\n    Returns\n    -------\n    human_mitosis: (512, 512) uint8 ndarray\n        Data of human cells undergoing mitosis taken during the preparation\n        of the manuscript in [1]_.\n\n    Notes\n    -----\n    Copyright David Root. Licensed under CC-0 [2]_.\n\n    References\n    ----------\n    .. [1] Moffat J, Grueneberg DA, Yang X, Kim SY, Kloepfer AM, Hinkle G,\n           Piqani B, Eisenhaure TM, Luo B, Grenier JK, Carpenter AE, Foo SY,\n           Stewart SA, Stockwell BR, Hacohen N, Hahn WC, Lander ES,\n           Sabatini DM, Root DE (2006) A lentiviral RNAi library for human and\n           mouse genes applied to an arrayed viral high-content screen. Cell,\n           124(6):1283-98 / :DOI: `10.1016/j.cell.2006.01.040` PMID 16564017\n\n    .. [2] GitHub licensing discussion\n           https://github.com/CellProfiler/examples/issues/41\n\n    '
    return _load('data/mitosis.tif')

def cell():
    if False:
        return 10
    'Cell floating in saline.\n\n    This is a quantitative phase image retrieved from a digital hologram using\n    the Python library ``qpformat``. The image shows a cell with high phase\n    value, above the background phase.\n\n    Because of a banding pattern artifact in the background, this image is a\n    good test of thresholding algorithms. The pixel spacing is 0.107 µm.\n\n    These data were part of a comparison between several refractive index\n    retrieval techniques for spherical objects as part of [1]_.\n\n    This image is CC0, dedicated to the public domain. You may copy, modify, or\n    distribute it without asking permission.\n\n    Returns\n    -------\n    cell : (660, 550) uint8 array\n        Image of a cell.\n\n    References\n    ----------\n    .. [1] Paul Müller, Mirjam Schürmann, Salvatore Girardo, Gheorghe Cojoc,\n           and Jochen Guck. "Accurate evaluation of size and refractive index\n           for spherical objects in quantitative phase imaging." Optics Express\n           26(8): 10729-10743 (2018). :DOI:`10.1364/OE.26.010729`\n    '
    return _load('data/cell.png')

def coins():
    if False:
        while True:
            i = 10
    'Greek coins from Pompeii.\n\n    This image shows several coins outlined against a gray background.\n    It is especially useful in, e.g. segmentation tests, where\n    individual objects need to be identified against a background.\n    The background shares enough grey levels with the coins that a\n    simple segmentation is not sufficient.\n\n    Notes\n    -----\n    This image was downloaded from the\n    `Brooklyn Museum Collection\n    <https://www.brooklynmuseum.org/opencollection/archives/image/51611>`__.\n\n    No known copyright restrictions.\n\n    Returns\n    -------\n    coins : (303, 384) uint8 ndarray\n        Coins image.\n    '
    return _load('data/coins.png')

def kidney():
    if False:
        print('Hello World!')
    'Mouse kidney tissue.\n\n    This biological tissue on a pre-prepared slide was imaged with confocal\n    fluorescence microscopy (Nikon C1 inverted microscope).\n    Image shape is (16, 512, 512, 3). That is 512x512 pixels in X-Y,\n    16 image slices in Z, and 3 color channels\n    (emission wavelengths 450nm, 515nm, and 605nm, respectively).\n    Real-space voxel size is 1.24 microns in X-Y, and 1.25 microns in Z.\n    Data type is unsigned 16-bit integers.\n\n    Notes\n    -----\n    This image was acquired by Genevieve Buckley at Monasoh Micro Imaging in\n    2018.\n    License: CC0\n\n    Returns\n    -------\n    kidney : (16, 512, 512, 3) uint16 ndarray\n        Kidney 3D multichannel image.\n    '
    return _load('data/kidney.tif')

def lily():
    if False:
        return 10
    'Lily of the valley plant stem.\n\n    This plant stem on a pre-prepared slide was imaged with confocal\n    fluorescence microscopy (Nikon C1 inverted microscope).\n    Image shape is (922, 922, 4). That is 922x922 pixels in X-Y,\n    with 4 color channels.\n    Real-space voxel size is 1.24 microns in X-Y.\n    Data type is unsigned 16-bit integers.\n\n    Notes\n    -----\n    This image was acquired by Genevieve Buckley at Monasoh Micro Imaging in\n    2018.\n    License: CC0\n\n    Returns\n    -------\n    lily : (922, 922, 4) uint16 ndarray\n        Lily 2D multichannel image.\n    '
    return _load('data/lily.tif')

def logo():
    if False:
        return 10
    'Scikit-image logo, a RGBA image.\n\n    Returns\n    -------\n    logo : (500, 500, 4) uint8 ndarray\n        Logo image.\n    '
    return _load('data/logo.png')

def microaneurysms():
    if False:
        return 10
    'Gray-level "microaneurysms" image.\n\n    Detail from an image of the retina (green channel).\n    The image is a crop of image 07_dr.JPG from the\n    High-Resolution Fundus (HRF) Image Database:\n    https://www5.cs.fau.de/research/data/fundus-images/\n\n    Notes\n    -----\n    No copyright restrictions. CC0 given by owner (Andreas Maier).\n\n    Returns\n    -------\n    microaneurysms : (102, 102) uint8 ndarray\n        Retina image with lesions.\n\n    References\n    ----------\n    .. [1] Budai, A., Bock, R, Maier, A., Hornegger, J.,\n           Michelson, G. (2013).  Robust Vessel Segmentation in Fundus\n           Images. International Journal of Biomedical Imaging, vol. 2013,\n           2013.\n           :DOI:`10.1155/2013/154860`\n    '
    return _load('data/microaneurysms.png')

def moon():
    if False:
        i = 10
        return i + 15
    'Surface of the moon.\n\n    This low-contrast image of the surface of the moon is useful for\n    illustrating histogram equalization and contrast stretching.\n\n    Returns\n    -------\n    moon : (512, 512) uint8 ndarray\n        Moon image.\n    '
    return _load('data/moon.png')

def page():
    if False:
        i = 10
        return i + 15
    'Scanned page.\n\n    This image of printed text is useful for demonstrations requiring uneven\n    background illumination.\n\n    Returns\n    -------\n    page : (191, 384) uint8 ndarray\n        Page image.\n    '
    return _load('data/page.png')

def horse():
    if False:
        while True:
            i = 10
    'Black and white silhouette of a horse.\n\n    This image was downloaded from\n    `openclipart <http://openclipart.org/detail/158377/horse-by-marauder>`\n\n    No copyright restrictions. CC0 given by owner (Andreas Preuss (marauder)).\n\n    Returns\n    -------\n    horse : (328, 400) bool ndarray\n        Horse image.\n    '
    return img_as_bool(_load('data/horse.png', as_gray=True))

def clock():
    if False:
        for i in range(10):
            print('nop')
    'Motion blurred clock.\n\n    This photograph of a wall clock was taken while moving the camera in an\n    approximately horizontal direction.  It may be used to illustrate\n    inverse filters and deconvolution.\n\n    Released into the public domain by the photographer (Stefan van der Walt).\n\n    Returns\n    -------\n    clock : (300, 400) uint8 ndarray\n        Clock image.\n    '
    return _load('data/clock_motion.png')

def immunohistochemistry():
    if False:
        print('Hello World!')
    'Immunohistochemical (IHC) staining with hematoxylin counterstaining.\n\n    This picture shows colonic glands where the IHC expression of FHL2 protein\n    is revealed with DAB. Hematoxylin counterstaining is applied to enhance the\n    negative parts of the tissue.\n\n    This image was acquired at the Center for Microscopy And Molecular Imaging\n    (CMMI).\n\n    No known copyright restrictions.\n\n    Returns\n    -------\n    immunohistochemistry : (512, 512, 3) uint8 ndarray\n        Immunohistochemistry image.\n    '
    return _load('data/ihc.png')

def chelsea():
    if False:
        return 10
    'Chelsea the cat.\n\n    An example with texture, prominent edges in horizontal and diagonal\n    directions, as well as features of differing scales.\n\n    Notes\n    -----\n    No copyright restrictions.  CC0 by the photographer (Stefan van der Walt).\n\n    Returns\n    -------\n    chelsea : (300, 451, 3) uint8 ndarray\n        Chelsea image.\n    '
    return _load('data/chelsea.png')
cat = chelsea

def coffee():
    if False:
        i = 10
        return i + 15
    'Coffee cup.\n\n    This photograph is courtesy of Pikolo Espresso Bar.\n    It contains several elliptical shapes as well as varying texture (smooth\n    porcelain to coarse wood grain).\n\n    Notes\n    -----\n    No copyright restrictions.  CC0 by the photographer (Rachel Michetti).\n\n    Returns\n    -------\n    coffee : (400, 600, 3) uint8 ndarray\n        Coffee image.\n    '
    return _load('data/coffee.png')

def hubble_deep_field():
    if False:
        return 10
    "Hubble eXtreme Deep Field.\n\n    This photograph contains the Hubble Telescope's farthest ever view of\n    the universe. It can be useful as an example for multi-scale\n    detection.\n\n    Notes\n    -----\n    This image was downloaded from\n    `HubbleSite\n    <http://hubblesite.org/newscenter/archive/releases/2012/37/image/a/>`__.\n\n    The image was captured by NASA and `may be freely used in the public domain\n    <http://www.nasa.gov/audience/formedia/features/MP_Photo_Guidelines.html>`_.\n\n    Returns\n    -------\n    hubble_deep_field : (872, 1000, 3) uint8 ndarray\n        Hubble deep field image.\n    "
    return _load('data/hubble_deep_field.jpg')

def retina():
    if False:
        i = 10
        return i + 15
    'Human retina.\n\n    This image of a retina is useful for demonstrations requiring circular\n    images.\n\n    Notes\n    -----\n    This image was downloaded from\n    `wikimedia <https://commons.wikimedia.org/wiki/File:Fundus_photograph_of_normal_left_eye.jpg>`.\n    This file is made available under the Creative Commons CC0 1.0 Universal\n    Public Domain Dedication.\n\n    References\n    ----------\n    .. [1] Häggström, Mikael (2014). "Medical gallery of Mikael Häggström 2014".\n           WikiJournal of Medicine 1 (2). :DOI:`10.15347/wjm/2014.008`.\n           ISSN 2002-4436. Public Domain\n\n    Returns\n    -------\n    retina : (1411, 1411, 3) uint8 ndarray\n        Retina image in RGB.\n    '
    return _load('data/retina.jpg')

def shepp_logan_phantom():
    if False:
        while True:
            i = 10
    'Shepp Logan Phantom.\n\n    References\n    ----------\n    .. [1] L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head\n           section," in IEEE Transactions on Nuclear Science, vol. 21,\n           no. 3, pp. 21-43, June 1974. :DOI:`10.1109/TNS.1974.6499235`\n\n    Returns\n    -------\n    phantom : (400, 400) float64 image\n        Image of the Shepp-Logan phantom in grayscale.\n    '
    return _load('data/phantom.png', as_gray=True)

def colorwheel():
    if False:
        i = 10
        return i + 15
    'Color Wheel.\n\n    Returns\n    -------\n    colorwheel : (370, 371, 3) uint8 image\n        A colorwheel.\n    '
    return _load('data/color.png')

def palisades_of_vogt():
    if False:
        print('Hello World!')
    'Return image sequence of in-vivo tissue showing the palisades of Vogt.\n\n    In the human eye, the palisades of Vogt are normal features of the corneal\n    limbus, which is the border between the cornea and the sclera (i.e., the\n    white of the eye).\n    In the image sequence, there are some dark spots due to the presence of\n    dust on the reference mirror.\n\n    Returns\n    -------\n    palisades_of_vogt: (60, 1440, 1440) uint16 ndarray\n\n    Notes\n    -----\n    See info under `in-vivo-cornea-spots.tif` at\n    https://gitlab.com/scikit-image/data/-/blob/master/README.md#data.\n\n    '
    return _load('data/palisades_of_vogt.tif')

def rocket():
    if False:
        for i in range(10):
            print('nop')
    "Launch photo of DSCOVR on Falcon 9 by SpaceX.\n\n    This is the launch photo of Falcon 9 carrying DSCOVR lifted off from\n    SpaceX's Launch Complex 40 at Cape Canaveral Air Force Station, FL.\n\n    Notes\n    -----\n    This image was downloaded from\n    `SpaceX Photos\n    <https://www.flickr.com/photos/spacexphotos/16511594820/in/photostream/>`__.\n\n    The image was captured by SpaceX and `released in the public domain\n    <http://arstechnica.com/tech-policy/2015/03/elon-musk-puts-spacex-photos-into-the-public-domain/>`_.\n\n    Returns\n    -------\n    rocket : (427, 640, 3) uint8 ndarray\n        Rocket image.\n    "
    return _load('data/rocket.jpg')

def stereo_motorcycle():
    if False:
        return 10
    'Rectified stereo image pair with ground-truth disparities.\n\n    The two images are rectified such that every pixel in the left image has\n    its corresponding pixel on the same scanline in the right image. That means\n    that both images are warped such that they have the same orientation but a\n    horizontal spatial offset (baseline). The ground-truth pixel offset in\n    column direction is specified by the included disparity map.\n\n    The two images are part of the Middlebury 2014 stereo benchmark. The\n    dataset was created by Nera Nesic, Porter Westling, Xi Wang, York Kitajima,\n    Greg Krathwohl, and Daniel Scharstein at Middlebury College. A detailed\n    description of the acquisition process can be found in [1]_.\n\n    The images included here are down-sampled versions of the default exposure\n    images in the benchmark. The images are down-sampled by a factor of 4 using\n    the function `skimage.transform.downscale_local_mean`. The calibration data\n    in the following and the included ground-truth disparity map are valid for\n    the down-sampled images::\n\n        Focal length:           994.978px\n        Principal point x:      311.193px\n        Principal point y:      254.877px\n        Principal point dx:      31.086px\n        Baseline:               193.001mm\n\n    Returns\n    -------\n    img_left : (500, 741, 3) uint8 ndarray\n        Left stereo image.\n    img_right : (500, 741, 3) uint8 ndarray\n        Right stereo image.\n    disp : (500, 741, 3) float ndarray\n        Ground-truth disparity map, where each value describes the offset in\n        column direction between corresponding pixels in the left and the right\n        stereo images. E.g. the corresponding pixel of\n        ``img_left[10, 10 + disp[10, 10]]`` is ``img_right[10, 10]``.\n        NaNs denote pixels in the left image that do not have ground-truth.\n\n    Notes\n    -----\n    The original resolution images, images with different exposure and\n    lighting, and ground-truth depth maps can be found at the Middlebury\n    website [2]_.\n\n    References\n    ----------\n    .. [1] D. Scharstein, H. Hirschmueller, Y. Kitajima, G. Krathwohl, N.\n           Nesic, X. Wang, and P. Westling. High-resolution stereo datasets\n           with subpixel-accurate ground truth. In German Conference on Pattern\n           Recognition (GCPR 2014), Muenster, Germany, September 2014.\n    .. [2] http://vision.middlebury.edu/stereo/data/scenes2014/\n\n    '
    filename = _fetch('data/motorcycle_disp.npz')
    disp = np.load(filename)['arr_0']
    return (_load('data/motorcycle_left.png'), _load('data/motorcycle_right.png'), disp)

def lfw_subset():
    if False:
        while True:
            i = 10
    'Subset of data from the LFW dataset.\n\n    This database is a subset of the LFW database containing:\n\n    * 100 faces\n    * 100 non-faces\n\n    The full dataset is available at [2]_.\n\n    Returns\n    -------\n    images : (200, 25, 25) uint8 ndarray\n        100 first images are faces and subsequent 100 are non-faces.\n\n    Notes\n    -----\n    The faces were randomly selected from the LFW dataset and the non-faces\n    were extracted from the background of the same dataset. The cropped ROIs\n    have been resized to a 25 x 25 pixels.\n\n    References\n    ----------\n    .. [1] Huang, G., Mattar, M., Lee, H., & Learned-Miller, E. G. (2012).\n           Learning to align from scratch. In Advances in Neural Information\n           Processing Systems (pp. 764-772).\n    .. [2] http://vis-www.cs.umass.edu/lfw/\n\n    '
    return np.load(_fetch('data/lfw_subset.npy'))

def skin():
    if False:
        while True:
            i = 10
    'Microscopy image of dermis and epidermis (skin layers).\n\n    Hematoxylin and eosin stained slide at 10x of normal epidermis and dermis\n    with a benign intradermal nevus.\n\n    Notes\n    -----\n    This image requires an Internet connection the first time it is called,\n    and to have the ``pooch`` package installed, in order to fetch the image\n    file from the scikit-image datasets repository.\n\n    The source of this image is\n    https://en.wikipedia.org/wiki/File:Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.JPG\n\n    The image was released in the public domain by its author Kilbad.\n\n    Returns\n    -------\n    skin : (960, 1280, 3) RGB image of uint8\n    '
    return _load('data/skin.jpg')

def nickel_solidification():
    if False:
        return 10
    'Image sequence of synchrotron x-radiographs showing the rapid\n    solidification of a nickel alloy sample.\n\n    Returns\n    -------\n    nickel_solidification: (11, 384, 512) uint16 ndarray\n\n    Notes\n    -----\n    See info under `nickel_solidification.tif` at\n    https://gitlab.com/scikit-image/data/-/blob/master/README.md#data.\n\n    '
    return _load('data/solidification.tif')

def protein_transport():
    if False:
        while True:
            i = 10
    'Microscopy image sequence with fluorescence tagging of proteins\n    re-localizing from the cytoplasmic area to the nuclear envelope.\n\n    Returns\n    -------\n    protein_transport: (15, 2, 180, 183) uint8 ndarray\n\n    Notes\n    -----\n    See info under `NPCsingleNucleus.tif` at\n    https://gitlab.com/scikit-image/data/-/blob/master/README.md#data.\n\n    '
    return _load('data/protein_transport.tif')

def brain():
    if False:
        for i in range(10):
            print('nop')
    'Subset of data from the University of North Carolina Volume Rendering\n    Test Data Set.\n\n    The full dataset is available at [1]_.\n\n    Returns\n    -------\n    image : (10, 256, 256) uint16 ndarray\n\n    Notes\n    -----\n    The 3D volume consists of 10 layers from the larger volume.\n\n    References\n    ----------\n    .. [1] https://graphics.stanford.edu/data/voldata/\n\n    '
    return _load('data/brain.tiff')

def vortex():
    if False:
        print('Hello World!')
    'Case B1 image pair from the first PIV challenge.\n\n    Returns\n    -------\n    image0, image1 : (512, 512) grayscale images\n        A pair of images featuring synthetic moving particles.\n\n    Notes\n    -----\n    This image was licensed as CC0 by its author, Prof. Koji Okamoto, with\n    thanks to Prof. Jun Sakakibara, who maintains the PIV Challenge site.\n\n    References\n    ----------\n    .. [1] Particle Image Velocimetry (PIV) Challenge site\n           http://pivchallenge.org\n    .. [2] 1st PIV challenge Case B: http://pivchallenge.org/pub/index.html#b\n    '
    return (_load('data/pivchallenge-B-B001_1.tif'), _load('data/pivchallenge-B-B001_2.tif'))