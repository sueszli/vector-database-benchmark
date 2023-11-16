import glob
import logging
import os
from reportlab import rl_config
'This module allows the mapping of some system-available TTF fonts to\nthe reportlab engine.\n\nThis file could be customized per distro (although most Linux/Unix ones)\nshould have the same filenames, only need the code below).\n\nDue to an awful configuration that ships with reportlab at many Linux\nand Ubuntu distros, we have to override the search path, too.\n'
_logger = logging.getLogger(__name__)
CustomTTFonts = []
TTFSearchPath = ['/usr/share/fonts/truetype', '/usr/share/fonts/dejavu', '/usr/share/fonts/liberation', '/usr/share/fonts/truetype/*', '/usr/local/share/fonts/usr/share/fonts/TTF/*', '/usr/share/fonts/TTF', '/usr/lib/openoffice/share/fonts/truetype/', '~/.fonts', '~/.local/share/fonts', '~/Library/Fonts', '/Library/Fonts', '/Network/Library/Fonts', '/System/Library/Fonts', 'c:/winnt/fonts', 'c:/windows/fonts']

def list_all_sysfonts():
    if False:
        print('Hello World!')
    '\n        This function returns list of font directories of system.\n    '
    filepath = []
    searchpath = list(set(TTFSearchPath + rl_config.TTFSearchPath))
    for dirname in searchpath:
        for filename in glob.glob(os.path.join(os.path.expanduser(dirname), '*.[Tt][Tt][FfCc]')):
            filepath.append(filename)
    return filepath

def SetCustomFonts(rmldoc):
    if False:
        for i in range(10):
            print('nop')
    ' Map some font names to the corresponding TTF fonts\n\n        The ttf font may not even have the same name, as in\n        Times -> Liberation Serif.\n        This function is called once per report, so it should\n        avoid system-wide processing (cache it, instead).\n    '
    for (family, font, filename, mode) in CustomTTFonts:
        if os.path.isabs(filename) and os.path.exists(filename):
            rmldoc.setTTFontMapping(family, font, filename, mode)
    return True