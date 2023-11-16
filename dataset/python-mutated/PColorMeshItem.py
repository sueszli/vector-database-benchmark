"""
Demonstrates very basic use of PColorMeshItem
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter
app = pg.mkQApp('PColorMesh Example')
win = pg.GraphicsLayoutWidget()
win.show()
win.setWindowTitle('pyqtgraph example: pColorMeshItem')
view_auto_scale = win.addPlot(0, 0, 1, 1, title='Auto-scaling colorscheme', enableMenu=False)
view_consistent_scale = win.addPlot(1, 0, 1, 1, title='Consistent colorscheme', enableMenu=False)
randomness = 5
xn = 50
yn = 40
x = np.repeat(np.arange(1, xn + 1), yn).reshape(xn, yn) + np.random.random((xn, yn)) * randomness
y = np.tile(np.arange(1, yn + 1), xn).reshape(xn, yn) + np.random.random((xn, yn)) * randomness
x.sort(axis=0)
y.sort(axis=0)
z = np.exp(-(x * xn) ** 2 / 1000)[:-1, :-1]
edgecolors = None
antialiasing = False
cmap = pg.colormap.get('viridis')
levels = (-2, 2)
pcmi_auto = pg.PColorMeshItem(edgecolors=edgecolors, antialiasing=antialiasing, colorMap=cmap, levels=levels, enableAutoLevels=True)
view_auto_scale.addItem(pcmi_auto)
bar = pg.ColorBarItem(label='Z value [arbitrary unit]', interactive=False, rounding=0.1)
bar.setImageItem([pcmi_auto])
win.addItem(bar, 0, 1, 1, 1)
pcmi_consistent = pg.PColorMeshItem(edgecolors=edgecolors, antialiasing=antialiasing, colorMap=cmap, levels=levels, enableAutoLevels=False)
view_consistent_scale.addItem(pcmi_consistent)
bar_static = pg.ColorBarItem(label='Z value [arbitrary unit]', interactive=True, rounding=0.1)
bar_static.setImageItem([pcmi_consistent])
win.addItem(bar_static, 1, 1, 1, 1)
textitem = pg.TextItem(anchor=(1, 0))
view_auto_scale.addItem(textitem)
wave_amplitude = 3
wave_speed = 0.3
wave_length = 10
color_speed = 0.3
color_noise_freq = 0.05
miny = np.min(y) - wave_amplitude
maxy = np.max(y) + wave_amplitude
view_auto_scale.setYRange(miny, maxy)
textitem.setPos(np.max(x), maxy)
textpos = None
i = 0

def updateData():
    if False:
        while True:
            i = 10
    global i
    global textpos
    color_noise = np.sin(i * 2 * np.pi * color_noise_freq)
    new_x = x
    new_y = y + wave_amplitude * np.cos(x / wave_length + i)
    new_z = np.exp(-(x - np.cos(i * color_speed) * xn) ** 2 / 1000)[:-1, :-1] + color_noise
    pcmi_auto.setData(new_x, new_y, new_z)
    pcmi_consistent.setData(new_x, new_y, new_z)
    i += wave_speed
    framecnt.update()
timer = QtCore.QTimer()
timer.timeout.connect(updateData)
timer.start()
framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: textitem.setText(f'{fps:.1f} fps'))
if __name__ == '__main__':
    pg.exec()