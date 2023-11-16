"""
======
Slider
======

In this example, sliders are used to control the frequency and amplitude of
a sine wave.

See :doc:`/gallery/widgets/slider_snap_demo` for an example of having
the ``Slider`` snap to discrete values.

See :doc:`/gallery/widgets/range_slider` for an example of using
a ``RangeSlider`` to define a range of values.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

def f(t, amplitude, frequency):
    if False:
        print('Hello World!')
    return amplitude * np.sin(2 * np.pi * frequency * t)
t = np.linspace(0, 1, 1000)
init_amplitude = 5
init_frequency = 3
(fig, ax) = plt.subplots()
(line,) = ax.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.set_xlabel('Time [s]')
fig.subplots_adjust(left=0.25, bottom=0.25)
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(ax=axfreq, label='Frequency [Hz]', valmin=0.1, valmax=30, valinit=init_frequency)
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(ax=axamp, label='Amplitude', valmin=0, valmax=10, valinit=init_amplitude, orientation='vertical')

def update(val):
    if False:
        i = 10
        return i + 15
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()
freq_slider.on_changed(update)
amp_slider.on_changed(update)
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    if False:
        return 10
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)
plt.show()