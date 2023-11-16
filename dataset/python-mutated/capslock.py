from __future__ import unicode_literals, division, absolute_import, print_function
try:
    import vim
except ImportError:
    vim = object()
from powerline.bindings.vim import vim_func_exists
from powerline.theme import requires_segment_info

@requires_segment_info
def capslock_indicator(pl, segment_info, text='CAPS'):
    if False:
        for i in range(10):
            print('nop')
    'Shows the indicator if tpope/vim-capslock plugin is enabled\n\n\t.. note::\n\t\tIn the current state plugin automatically disables itself when leaving \n\t\tinsert mode. So trying to use this segment not in insert or replace \n\t\tmodes is useless.\n\n\t:param str text:\n\t\tString to show when software capslock presented by this plugin is \n\t\tactive.\n\t'
    if not vim_func_exists('CapsLockStatusline'):
        return None
    return text if vim.eval('CapsLockStatusline()') else None