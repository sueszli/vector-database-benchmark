import json
import vim
from powerline.lib.unicode import u
_powerline_old_render = powerline.render

def _powerline_test_render_function(*args, **kwargs):
    if False:
        return 10
    ret = _powerline_old_render(*args, **kwargs)
    vim.eval('add(g:statusline_values, %s)' % json.dumps(u(ret)))
    return ret
powerline.render = _powerline_test_render_function