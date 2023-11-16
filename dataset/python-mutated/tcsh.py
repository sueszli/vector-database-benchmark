from __future__ import unicode_literals, division, absolute_import, print_function
from powerline.renderers.shell.zsh import ZshPromptRenderer

class TcshPromptRenderer(ZshPromptRenderer):
    """Powerline tcsh prompt segment renderer."""
    character_translations = ZshPromptRenderer.character_translations.copy()
    character_translations[ord('%')] = '%%'
    character_translations[ord('\\')] = '\\\\'
    character_translations[ord('^')] = '\\^'
    character_translations[ord('!')] = '\\!'

    def do_render(self, **kwargs):
        if False:
            while True:
                i = 10
        ret = super(TcshPromptRenderer, self).do_render(**kwargs)
        nbsp = self.character_translations.get(ord(' '), ' ')
        end = self.hlstyle()
        assert not ret or ret.endswith(end)
        if ret.endswith(nbsp + end):
            ret = ret[:-(len(nbsp) + len(end))] + end + nbsp
        else:
            ret += nbsp
        return ret
renderer = TcshPromptRenderer