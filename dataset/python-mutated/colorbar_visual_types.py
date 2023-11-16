"""Using Colorbars with the Canvas to demo different types"""
from vispy import app
from vispy import gloo
from vispy.visuals import ColorBarVisual
from vispy.color import get_colormap
colormap = get_colormap('viridis')

def style_colorbar(colorbar):
    if False:
        print('Hello World!')
    colorbar.label.font_size = 18
    colorbar.label.color = 'black'
    colorbar.clim = (0, 10)
    colorbar.ticks[0].font_size = 14
    colorbar.ticks[1].font_size = 14
    colorbar.ticks[0].color = 'black'
    colorbar.ticks[1].color = 'black'
    colorbar.border_width = 1
    colorbar.border_color = 'black'
    return colorbar

def get_left_orientation_bar():
    if False:
        i = 10
        return i + 15
    pos = (50, 300)
    size = (400, 10)
    colorbar = ColorBarVisual(pos=pos, size=size, label='orientation left', cmap=colormap, orientation='left')
    return style_colorbar(colorbar)

def get_right_orientation_bar():
    if False:
        for i in range(10):
            print('nop')
    pos = (200, 300)
    size = (400, 10)
    colorbar = ColorBarVisual(pos=pos, size=size, label='orientation right', cmap=colormap, orientation='right')
    return style_colorbar(colorbar)

def get_top_orientation_bar():
    if False:
        while True:
            i = 10
    pos = (600, 400)
    size = (300, 10)
    colorbar = ColorBarVisual(pos=pos, size=size, label='orientation top', cmap=colormap, orientation='top')
    return style_colorbar(colorbar)

def get_bottom_orientation_bar():
    if False:
        for i in range(10):
            print('nop')
    pos = (600, 150)
    size = (300, 10)
    colorbar = ColorBarVisual(pos=pos, size=size, label='orientation bottom', cmap=colormap, orientation='bottom')
    return style_colorbar(colorbar)

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, size=(800, 600), keys='interactive')
        self.bars = []
        self.bars.append(get_left_orientation_bar())
        self.bars.append(get_right_orientation_bar())
        self.bars.append(get_top_orientation_bar())
        self.bars.append(get_bottom_orientation_bar())
        self.show()

    def on_resize(self, event):
        if False:
            i = 10
            return i + 15
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        for bar in self.bars:
            bar.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear(color='white')
        for bar in self.bars:
            bar.draw()
if __name__ == '__main__':
    win = Canvas()
    app.run()