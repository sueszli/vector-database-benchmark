from __future__ import unicode_literals, division, absolute_import, print_function
from powerline import Powerline
from powerline.lib.dict import mergedicts

class LemonbarPowerline(Powerline):

    def init(self):
        if False:
            return 10
        super(LemonbarPowerline, self).init(ext='wm', renderer_module='lemonbar')
    get_encoding = staticmethod(lambda : 'utf-8')

    def get_local_themes(self, local_themes):
        if False:
            for i in range(10):
                print('nop')
        if not local_themes:
            return {}
        return dict(((key, {'config': self.load_theme_config(val)}) for (key, val) in local_themes.items()))