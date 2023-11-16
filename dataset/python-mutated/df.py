import os
from libqtile.widget import base

class DF(base.ThreadPoolText):
    """Disk Free Widget

    By default the widget only displays if the space is less than warn_space.
    """
    defaults = [('partition', '/', 'the partition to check space'), ('warn_color', 'ff0000', 'Warning color'), ('warn_space', 2, 'Warning space in scale defined by the ``measure`` option.'), ('visible_on_warn', True, 'Only display if warning'), ('measure', 'G', 'Measurement (G, M, B)'), ('format', '{p} ({uf}{m}|{r:.0f}%)', 'String format (p: partition, s: size, f: free space, uf: user free space, m: measure, r: ratio (uf/s))'), ('update_interval', 60, 'The update interval.')]
    measures = {'G': 1024 * 1024 * 1024, 'M': 1024 * 1024, 'B': 1024}

    def __init__(self, **config):
        if False:
            print('Hello World!')
        base.ThreadPoolText.__init__(self, '', **config)
        self.add_defaults(DF.defaults)
        self.user_free = 0
        self.calc = self.measures[self.measure]

    def draw(self):
        if False:
            i = 10
            return i + 15
        if self.user_free <= self.warn_space:
            self.layout.colour = self.warn_color
        else:
            self.layout.colour = self.foreground
        base.ThreadPoolText.draw(self)

    def poll(self):
        if False:
            print('Hello World!')
        statvfs = os.statvfs(self.partition)
        size = statvfs.f_frsize * statvfs.f_blocks // self.calc
        free = statvfs.f_frsize * statvfs.f_bfree // self.calc
        self.user_free = statvfs.f_frsize * statvfs.f_bavail // self.calc
        if self.visible_on_warn and self.user_free >= self.warn_space:
            text = ''
        else:
            text = self.format.format(p=self.partition, s=size, f=free, uf=self.user_free, m=self.measure, r=(size - self.user_free) / size * 100)
        return text