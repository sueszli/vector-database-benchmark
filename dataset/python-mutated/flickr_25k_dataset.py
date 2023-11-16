import os
from tensorlayer import logging, visualize
from tensorlayer.files.utils import del_file, folder_exists, load_file_list, maybe_download_and_extract, natural_keys, read_file
__all__ = ['load_flickr25k_dataset']

def load_flickr25k_dataset(tag='sky', path='data', n_threads=50, printable=False):
    if False:
        for i in range(10):
            print('nop')
    "Load Flickr25K dataset.\n\n    Returns a list of images by a given tag from Flick25k dataset,\n    it will download Flickr25k from `the official website <http://press.liacs.nl/mirflickr/mirdownload.html>`__\n    at the first time you use it.\n\n    Parameters\n    ------------\n    tag : str or None\n        What images to return.\n            - If you want to get images with tag, use string like 'dog', 'red', see `Flickr Search <https://www.flickr.com/search/>`__.\n            - If you want to get all images, set to ``None``.\n\n    path : str\n        The path that the data is downloaded to, defaults is ``data/flickr25k/``.\n    n_threads : int\n        The number of thread to read image.\n    printable : boolean\n        Whether to print infomation when reading images, default is ``False``.\n\n    Examples\n    -----------\n    Get images with tag of sky\n\n    >>> images = tl.files.load_flickr25k_dataset(tag='sky')\n\n    Get all images\n\n    >>> images = tl.files.load_flickr25k_dataset(tag=None, n_threads=100, printable=True)\n\n    "
    path = os.path.join(path, 'flickr25k')
    filename = 'mirflickr25k.zip'
    url = 'http://press.liacs.nl/mirflickr/mirflickr25k/'
    if folder_exists(os.path.join(path, 'mirflickr')) is False:
        logging.info('[*] Flickr25k is nonexistent in {}'.format(path))
        maybe_download_and_extract(filename, path, url, extract=True)
        del_file(os.path.join(path, filename))
    folder_imgs = os.path.join(path, 'mirflickr')
    path_imgs = load_file_list(path=folder_imgs, regx='\\.jpg', printable=False)
    path_imgs.sort(key=natural_keys)
    folder_tags = os.path.join(path, 'mirflickr', 'meta', 'tags')
    path_tags = load_file_list(path=folder_tags, regx='\\.txt', printable=False)
    path_tags.sort(key=natural_keys)
    if tag is None:
        logging.info('[Flickr25k] reading all images')
    else:
        logging.info('[Flickr25k] reading images with tag: {}'.format(tag))
    images_list = []
    for (idx, _v) in enumerate(path_tags):
        tags = read_file(os.path.join(folder_tags, path_tags[idx])).split('\n')
        if tag is None or tag in tags:
            images_list.append(path_imgs[idx])
    images = visualize.read_images(images_list, folder_imgs, n_threads=n_threads, printable=printable)
    return images