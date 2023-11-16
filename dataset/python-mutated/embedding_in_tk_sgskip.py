"""
===============
Embedding in Tk
===============

"""
import tkinter
import numpy as np
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
root = tkinter.Tk()
root.wm_title('Embedding in Tk')
fig = Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, 0.01)
ax = fig.add_subplot()
(line,) = ax.plot(t, 2 * np.sin(2 * np.pi * t))
ax.set_xlabel('time [s]')
ax.set_ylabel('f(t)')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
canvas.mpl_connect('key_press_event', lambda event: print(f'you pressed {event.key}'))
canvas.mpl_connect('key_press_event', key_press_handler)
button_quit = tkinter.Button(master=root, text='Quit', command=root.destroy)

def update_frequency(new_val):
    if False:
        while True:
            i = 10
    f = float(new_val)
    y = 2 * np.sin(2 * np.pi * f * t)
    line.set_data(t, y)
    canvas.draw()
slider_update = tkinter.Scale(root, from_=1, to=5, orient=tkinter.HORIZONTAL, command=update_frequency, label='Frequency [Hz]')
button_quit.pack(side=tkinter.BOTTOM)
slider_update.pack(side=tkinter.BOTTOM)
toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
tkinter.mainloop()