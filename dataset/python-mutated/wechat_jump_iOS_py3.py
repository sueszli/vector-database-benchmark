import time
import wda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
time_coefficient = 0.0012
VERSION = '1.1.4'
c = wda.Client()
s = c.session()

def pull_screenshot():
    if False:
        print('Hello World!')
    c.screenshot('autojump.png')

def jump(distance):
    if False:
        i = 10
        return i + 15
    press_time = distance * time_coefficient
    press_time = press_time
    print('press_time = ', press_time)
    s.tap_hold(200, 200, press_time)
fig = plt.figure()
pull_screenshot()
img = np.array(Image.open('autojump.png'))
im = plt.imshow(img, animated=True)
update = True
click_count = 0
cor = []

def update_data():
    if False:
        for i in range(10):
            print('nop')
    return np.array(Image.open('autojump.png'))

def updatefig(*args):
    if False:
        for i in range(10):
            print('nop')
    global update
    if update:
        time.sleep(1)
        pull_screenshot()
        im.set_array(update_data())
        update = False
    return (im,)

def on_click(event):
    if False:
        print('Hello World!')
    global update
    global ix, iy
    global click_count
    global cor
    (ix, iy) = (event.xdata, event.ydata)
    coords = [(ix, iy)]
    print('now = ', coords)
    cor.append(coords)
    click_count += 1
    if click_count > 1:
        click_count = 0
        cor1 = cor.pop()
        cor2 = cor.pop()
        distance = (cor1[0][0] - cor2[0][0]) ** 2 + (cor1[0][1] - cor2[0][1]) ** 2
        distance = distance ** 0.5
        print('distance = ', distance)
        jump(distance)
        update = True
fig.canvas.mpl_connect('button_press_event', on_click)
ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()