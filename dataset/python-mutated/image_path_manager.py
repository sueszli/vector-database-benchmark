import os
import os.path as osp
import warnings
from spyder.config.base import get_module_data_path
from spyder.config.gui import is_dark_interface

class ImagePathManager:
    """Manager of the image path in the project."""

    def __init__(self):
        if False:
            print('Hello World!')
        'Initialize main path with all the images.'
        self.IMG_PATH = {}
        self.add_image_path(get_module_data_path('spyder', relpath='images'))
        self.default = 'not_found'

    def add_image_path(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Add path to the image path list.'
        if not osp.isdir(path):
            return
        for (dirpath, __, _filenames) in os.walk(path):
            if is_dark_interface() and osp.basename(dirpath) == 'light':
                continue
            elif not is_dark_interface() and osp.basename(dirpath) == 'dark':
                continue
            for filename in _filenames:
                if filename.startswith('.'):
                    continue
                (name, __) = osp.splitext(osp.basename(filename))
                complete_path = osp.join(dirpath, filename)
                if name in self.IMG_PATH:
                    warnings.warn(f'The icon located in {complete_path} is overriding the existing {name}')
                self.IMG_PATH[name] = complete_path

    def get_image_path(self, name):
        if False:
            print('Hello World!')
        'Get path of the image given its name.'
        try:
            act_image = self.IMG_PATH[name]
            if osp.isfile(act_image):
                return osp.abspath(act_image)
        except KeyError:
            return osp.abspath(self.IMG_PATH[self.default])
IMAGE_PATH_MANAGER = ImagePathManager()

def get_image_path(name):
    if False:
        while True:
            i = 10
    'Return absolute image path.'
    return IMAGE_PATH_MANAGER.get_image_path(name)