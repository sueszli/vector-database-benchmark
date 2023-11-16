import os
from tensorlayer import logging
from tensorlayer.files.utils import maybe_download_and_extract
__all__ = ['load_nietzsche_dataset']

def load_nietzsche_dataset(path='data'):
    if False:
        print('Hello World!')
    'Load Nietzsche dataset.\n\n    Parameters\n    ----------\n    path : str\n        The path that the data is downloaded to, defaults is ``data/nietzsche/``.\n\n    Returns\n    --------\n    str\n        The content.\n\n    Examples\n    --------\n    >>> see tutorial_generate_text.py\n    >>> words = tl.files.load_nietzsche_dataset()\n    >>> words = basic_clean_str(words)\n    >>> words = words.split()\n\n    '
    logging.info('Load or Download nietzsche dataset > {}'.format(path))
    path = os.path.join(path, 'nietzsche')
    filename = 'nietzsche.txt'
    url = 'https://s3.amazonaws.com/text-datasets/'
    filepath = maybe_download_and_extract(filename, path, url)
    with open(filepath, 'r') as f:
        words = f.read()
        return words