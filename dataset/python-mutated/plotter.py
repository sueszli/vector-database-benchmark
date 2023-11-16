import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import figure
from pathlib import Path, PurePath
from typing import Dict, Union, List
import numpy as np
from PIL import Image

def _formatter(val: Union[int, np.float32]):
    if False:
        print('Hello World!')
    '\n    For printing floats only upto 3rd precision. Ints are unchanged.\n    '
    if isinstance(val, np.float32):
        return f'{val:.3f}'
    else:
        return val

def _plot_images(image_dir: PurePath, orig: str, image_list: List, scores: bool=False, outfile: str=None) -> None:
    if False:
        print('Hello World!')
    '\n    Plotting function for plot_duplicates() defined below.\n\n    Args:\n        image_dir: image directory where all files in duplicate_map are present.\n        orig: filename for which duplicates are to be plotted.\n        image_list: List of duplicate filenames, could also be with scores (filename, score).\n        scores: Whether only filenames are present in the image_list or scores as well.\n        outfile:  Name of the file to save the plot.\n    '
    n_ims = len(image_list)
    ncols = 4
    nrows = int(np.ceil(n_ims / ncols)) + 1
    fig = figure.Figure(figsize=(10, 14))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    ax = plt.subplot(gs[0, 1:3])
    ax.imshow(Image.open(image_dir / orig))
    ax.set_title('Original Image: {}'.format(orig))
    ax.axis('off')
    for i in range(0, n_ims):
        row_num = i // ncols + 1
        col_num = i % ncols
        ax = plt.subplot(gs[row_num, col_num])
        if scores:
            ax.imshow(Image.open(image_dir / image_list[i][0]))
            val = _formatter(image_list[i][1])
            title = ' '.join([image_list[i][0], f'({val})'])
        else:
            ax.imshow(Image.open(image_dir / image_list[i]))
            title = image_list[i]
        ax.set_title(title, fontsize=6)
        ax.axis('off')
    gs.tight_layout(fig)
    if outfile:
        plt.savefig(outfile)
    plt.show()
    plt.close()

def _validate_args(image_dir: Union[PurePath, str], duplicate_map: Dict, filename: str) -> PurePath:
    if False:
        return 10
    'Argument validator for plot_duplicates() defined below.\n    Return PurePath to the image directory'
    image_dir = Path(image_dir)
    assert image_dir.is_dir(), 'Provided image directory does not exist! Please provide the image directory where all files are present!'
    if not isinstance(duplicate_map, dict):
        raise ValueError('Please provide a valid Duplicate map!')
    if filename not in duplicate_map.keys():
        raise ValueError('Please provide a valid filename present as a key in the duplicate_map!')
    return image_dir

def plot_duplicates(image_dir: Union[PurePath, str], duplicate_map: Dict, filename: str, outfile: str=None) -> None:
    if False:
        return 10
    "\n    Given filename for an image, plot duplicates along with the original image using the duplicate map obtained using\n    find_duplicates method.\n\n    Args:\n        image_dir: image directory where all files in duplicate_map are present.\n        duplicate_map: mapping of filename to found duplicates (could be with or without scores).\n        filename: Name of the file for which duplicates are to be plotted, must be a key in the duplicate_map.\n        dictionary.\n        outfile: Optional, name of the file to save the plot. Default is None.\n\n    Example:\n    ```\n        from imagededup.utils import plot_duplicates\n        plot_duplicates(image_dir='path/to/image/directory',\n                        duplicate_map=duplicate_map,\n                        filename='path/to/image.jpg')\n    ```\n    "
    image_dir = _validate_args(image_dir=image_dir, duplicate_map=duplicate_map, filename=filename)
    retrieved = duplicate_map[filename]
    assert len(retrieved) != 0, 'Provided filename has no duplicates!'
    if isinstance(retrieved[0], tuple):
        _plot_images(image_dir=image_dir, orig=filename, image_list=retrieved, scores=True, outfile=outfile)
    else:
        _plot_images(image_dir=image_dir, orig=filename, image_list=retrieved, scores=False, outfile=outfile)