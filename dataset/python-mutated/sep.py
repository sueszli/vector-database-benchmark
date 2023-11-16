from libqtile.widget import base

class Sep(base._Widget):
    """A visible widget separator"""
    orientations = base.ORIENTATION_BOTH
    defaults = [('padding', 2, 'Padding on either side of separator.'), ('linewidth', 1, 'Width of separator line.'), ('foreground', '888888', 'Separator line colour.'), ('size_percent', 80, 'Size as a percentage of bar size (0-100).')]

    def __init__(self, **config):
        if False:
            print('Hello World!')
        length = config.get('padding', 2) * 2 + config.get('linewidth', 1)
        base._Widget.__init__(self, length, **config)
        self.add_defaults(Sep.defaults)
        self.length = self.padding + self.linewidth

    def draw(self):
        if False:
            print('Hello World!')
        self.drawer.clear(self.background or self.bar.background)
        if self.bar.horizontal:
            margin_top = self.bar.height / float(100) * (100 - self.size_percent) / 2.0
            self.drawer.draw_vbar(self.foreground, float(self.length) / 2, margin_top, self.bar.height - margin_top, linewidth=self.linewidth)
            self.drawer.draw(offsetx=self.offset, offsety=self.offsety, width=self.length)
        else:
            margin_left = self.bar.width / float(100) * (100 - self.size_percent) / 2.0
            self.drawer.draw_hbar(self.foreground, margin_left, self.bar.width - margin_left, float(self.length) / 2, linewidth=self.linewidth)
            self.drawer.draw(offsety=self.offset, offsetx=self.offsetx, height=self.length)