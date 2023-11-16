import os
import tempfile
from manimlib.config import get_custom_config
from manimlib.config import get_manim_dir
CUSTOMIZATION = {}

def get_customization():
    if False:
        while True:
            i = 10
    if not CUSTOMIZATION:
        CUSTOMIZATION.update(get_custom_config())
        directories = CUSTOMIZATION['directories']
        if not directories['temporary_storage']:
            directories['temporary_storage'] = tempfile.gettempdir()
        directories['shaders'] = os.path.join(get_manim_dir(), 'manimlib', 'shaders')
    return CUSTOMIZATION