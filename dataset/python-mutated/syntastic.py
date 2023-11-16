from __future__ import unicode_literals, division, absolute_import, print_function
try:
    import vim
except ImportError:
    vim = object()
from powerline.segments.vim import window_cached
from powerline.bindings.vim import vim_global_exists

@window_cached
def syntastic(pl, err_format='ERR: \ue0a1 {first_line} ({num}) ', warn_format='WARN: \ue0a1 {first_line} ({num}) '):
    if False:
        while True:
            i = 10
    'Show whether syntastic has found any errors or warnings\n\n\t:param str err_format:\n\t\tFormat string for errors.\n\n\t:param str warn_format:\n\t\tFormat string for warnings.\n\n\tHighlight groups used: ``syntastic:warning`` or ``warning``, ``syntastic:error`` or ``error``.\n\t'
    if not vim_global_exists('SyntasticLoclist'):
        return None
    has_errors = int(vim.eval('g:SyntasticLoclist.current().hasErrorsOrWarningsToDisplay()'))
    if not has_errors:
        return
    errors = vim.eval('g:SyntasticLoclist.current().errors()')
    warnings = vim.eval('g:SyntasticLoclist.current().warnings()')
    segments = []
    if errors:
        segments.append({'contents': err_format.format(first_line=errors[0]['lnum'], num=len(errors)), 'highlight_groups': ['syntastic:error', 'error']})
    if warnings:
        segments.append({'contents': warn_format.format(first_line=warnings[0]['lnum'], num=len(warnings)), 'highlight_groups': ['syntastic:warning', 'warning']})
    return segments