"""
Show border
===========

Shows widget's border.

The idea was taken from
http://robertour.com/2013/10/02/easy-way-debugging-kivy-interfaces/
"""
__all__ = ('start', 'stop')
from kivy.lang import Builder
KV_CODE = '\n<Widget>:\n    canvas.after:\n        Color:\n            rgba: 1, 1, 1, 1\n        Line:\n            rectangle: self.x + 1, self.y + 1, self.width - 1, self.height - 1\n            dash_offset: 5\n            dash_length: 3\n'

def start(win, ctx):
    if False:
        print('Hello World!')
    Builder.load_string(KV_CODE, filename=__file__)

def stop(win, ctx):
    if False:
        print('Hello World!')
    Builder.unload_file(__file__)