"""
binaries hook for pygame seems to be required for pygame 2.0 Windows.
Otherwise some essential DLLs will not be transferred to the exe.

And also put hooks for datas, resources that pygame uses, to work
correctly with pyinstaller
"""
import os
import platform
from pygame import __file__ as pygame_main_file
pygame_folder = os.path.dirname(os.path.abspath(pygame_main_file))
datas = []

def _append_to_datas(file_path):
    if False:
        return 10
    res_path = os.path.join(pygame_folder, file_path)
    if os.path.exists(res_path):
        datas.append((res_path, 'pygame'))
_append_to_datas('freesansbold.ttf')
if platform.system() == 'Darwin':
    _append_to_datas('pygame_icon_mac.bmp')
else:
    _append_to_datas('pygame_icon.bmp')
if platform.system() == 'Windows':
    from PyInstaller.utils.hooks import collect_dynamic_libs
    pre_binaries = collect_dynamic_libs('pygame')
    binaries = []
    for b in pre_binaries:
        (binary, location) = b
        binaries.append((binary, '.'))