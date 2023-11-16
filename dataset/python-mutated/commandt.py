from __future__ import unicode_literals, division, absolute_import, print_function
import os
from powerline.bindings.vim import vim_getbufoption, buffer_name

def commandt(matcher_info):
    if False:
        print('Hello World!')
    name = buffer_name(matcher_info)
    return vim_getbufoption(matcher_info, 'filetype') == 'command-t' or (name and os.path.basename(name) == b'GoToFile')