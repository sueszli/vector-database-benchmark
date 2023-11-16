from libqtile.widget.base import _TextBox

class ConfigErrorWidget(_TextBox):

    def __init__(self, **config):
        if False:
            while True:
                i = 10
        _TextBox.__init__(self, **config)
        self.class_name = self.widget.__class__.__name__
        self.text = 'Widget crashed: {} (click to hide)'.format(self.class_name)
        self.add_callbacks({'Button1': self._hide})

    def _hide(self):
        if False:
            return 10
        self.text = ''
        self.bar.draw()