import sys
from vispy import app
from vispy.gloo import clear
canvas = app.Canvas(size=(512, 512), title='Do nothing benchmark (vispy)', keys='interactive')

@canvas.connect
def on_draw(event):
    if False:
        for i in range(10):
            print('nop')
    clear(color=True, depth=True)
    canvas.update()
if __name__ == '__main__':
    canvas.show()
    canvas.measure_fps()
    if sys.flags.interactive == 0:
        app.run()