"""
================
pyplot with GTK4
================

An example of how to use pyplot to manage your figure windows, but modify the
GUI by accessing the underlying GTK widgets.
"""
import matplotlib
matplotlib.use('GTK4Agg')
import gi
import matplotlib.pyplot as plt
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk
(fig, ax) = plt.subplots()
ax.plot([1, 2, 3], 'ro-', label='easy as 1 2 3')
ax.plot([1, 4, 9], 'gs--', label='easy as 1 2 3 squared')
ax.legend()
manager = fig.canvas.manager
toolbar = manager.toolbar
vbox = manager.vbox
button = Gtk.Button(label='Click me')
button.connect('clicked', lambda button: print('hi mom'))
button.set_tooltip_text('Click me for fun and profit')
toolbar.append(button)
label = Gtk.Label()
label.set_markup('Drag mouse over axes for position')
vbox.insert_child_after(label, fig.canvas)

def update(event):
    if False:
        return 10
    if event.xdata is None:
        label.set_markup('Drag mouse over axes for position')
    else:
        label.set_markup(f'<span color="#ef0000">x,y=({event.xdata}, {event.ydata})</span>')
fig.canvas.mpl_connect('motion_notify_event', update)
plt.show()