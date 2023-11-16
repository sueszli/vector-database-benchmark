"""
Optical system design demo
"""
import numpy as np
from optics import Mirror, Ray, Tracer, Lens
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore
app = pg.mkQApp('Optics Demo')
w = pg.GraphicsLayoutWidget(show=True, border=0.5)
w.resize(1000, 900)
w.show()
view = w.addViewBox()
view.setAspectLocked()
view.setRange(pg.QtCore.QRectF(-50, -30, 100, 100))
optics = []
rays = []
m1 = Mirror(r1=-100, pos=(5, 0), d=5, angle=-15)
optics.append(m1)
m2 = Mirror(r1=-70, pos=(-40, 30), d=6, angle=180 - 15)
optics.append(m2)
allRays = []
for y in np.linspace(-10, 10, 21):
    r = Ray(start=Point(-100, y))
    view.addItem(r)
    allRays.append(r)
for o in optics:
    view.addItem(o)
t1 = Tracer(allRays, optics)
optics = []
view = w.addViewBox()
view.setAspectLocked()
view.setRange(pg.QtCore.QRectF(-10, -50, 90, 60))
optics = []
rays = []
l1 = Lens(r1=20, r2=20, d=10, angle=8, glass='Corning7980')
optics.append(l1)
allRays = []
for wl in np.linspace(355, 1040, 25):
    for y in [10]:
        r = Ray(start=Point(-100, y), wl=wl)
        view.addItem(r)
        allRays.append(r)
for o in optics:
    view.addItem(o)
t2 = Tracer(allRays, optics)
w.nextRow()
view = w.addViewBox(colspan=2)
optics = []
view.setRange(QtCore.QRectF(200, -50, 500, 200))
scanx = 250
scany = 20
m1 = Mirror(dia=4.2, d=0.001, pos=(scanx, 0), angle=315)
m2 = Mirror(dia=8.4, d=0.001, pos=(scanx, scany), angle=135)
l3 = Lens(r1=23.0, r2=0, d=5.8, pos=(scanx + 50, scany), glass='Corning7980')
l4 = Lens(r1=0, r2=69.0, d=3.2, pos=(scanx + 250, scany), glass='Corning7980')
obj = Lens(r1=15, r2=15, d=10, dia=8, pos=(scanx + 400, scany), glass='Corning7980')
IROptics = [m1, m2, l3, l4, obj]
scanx = 250
scany = 30
m1a = Mirror(dia=4.2, d=0.001, pos=(scanx, 2 * scany), angle=315)
m2a = Mirror(dia=8.4, d=0.001, pos=(scanx, 3 * scany), angle=135)
l3a = Lens(r1=46, r2=0, d=3.8, pos=(scanx + 50, 3 * scany), glass='Corning7980')
l4a = Lens(r1=0, r2=46, d=3.8, pos=(scanx + 250, 3 * scany), glass='Corning7980')
obja = Lens(r1=15, r2=15, d=10, dia=8, pos=(scanx + 400, 3 * scany), glass='Corning7980')
IROptics2 = [m1a, m2a, l3a, l4a, obja]
for o in set(IROptics + IROptics2):
    view.addItem(o)
IRRays = []
IRRays2 = []
for dy in [-0.4, -0.15, 0, 0.15, 0.4]:
    IRRays.append(Ray(start=Point(-50, dy), dir=(1, 0), wl=780))
    IRRays2.append(Ray(start=Point(-50, dy + 2 * scany), dir=(1, 0), wl=780))
for r in set(IRRays + IRRays2):
    view.addItem(r)
IRTracer = Tracer(IRRays, IROptics)
IRTracer2 = Tracer(IRRays2, IROptics2)
phase = 0.0

def update():
    if False:
        i = 10
        return i + 15
    global phase
    if phase % (8 * np.pi) > 4 * np.pi:
        m1['angle'] = 315 + 1.5 * np.sin(phase)
        m1a['angle'] = 315 + 1.5 * np.sin(phase)
    else:
        m2['angle'] = 135 + 1.5 * np.sin(phase)
        m2a['angle'] = 135 + 1.5 * np.sin(phase)
    phase += 0.2
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(40)
if __name__ == '__main__':
    pg.exec()