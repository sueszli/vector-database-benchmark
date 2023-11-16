from __future__ import unicode_literals, division, absolute_import, print_function
import os
from powerline.bindings.vim import buffer_name

def gundo(matcher_info):
    if False:
        return 10
    name = buffer_name(matcher_info)
    return name and os.path.basename(name) == b'__Gundo__'

def gundo_preview(matcher_info):
    if False:
        while True:
            i = 10
    name = buffer_name(matcher_info)
    return name and os.path.basename(name) == b'__Gundo_Preview__'