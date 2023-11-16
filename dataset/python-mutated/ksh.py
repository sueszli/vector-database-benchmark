from __future__ import unicode_literals, division, absolute_import, print_function
from powerline.renderers.shell import ShellRenderer
ESCAPE_CHAR = '\x01'

class KshPromptRenderer(ShellRenderer):
    """Powerline bash prompt segment renderer."""
    escape_hl_start = '\x01'
    escape_hl_end = '\x01'

    def render(self, *args, **kwargs):
        if False:
            return 10
        return '\x01\r' + super(KshPromptRenderer, self).render(*args, **kwargs)
renderer = KshPromptRenderer