"""
================
pyplot with GTK3
================

An example of how to use pyplot to manage your figure windows, but modify the
GUI by accessing the underlying GTK widgets.
"""
import matplotlib
matplotlib.use('GTK3Agg')
import gi
import matplotlib.pyplot as plt
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
(fig, ax) = plt.subplots()
ax.plot([1, 2, 3], 'ro-', label='easy as 1 2 3')
ax.plot([1, 4, 9], 'gs--', label='easy as 1 2 3 squared')
ax.legend()
manager = fig.canvas.manager
toolbar = manager.toolbar
vbox = manager.vbox
button = Gtk.Button(label='Click me')
button.show()
button.connect('clicked', lambda button: print('hi mom'))
toolitem = Gtk.ToolItem()
toolitem.show()
toolitem.set_tooltip_text('Click me for fun and profit')
toolitem.add(button)
pos = 8
toolbar.insert(toolitem, pos)
label = Gtk.Label()
label.set_markup('Drag mouse over axes for position')
label.show()
vbox.pack_start(label, False, False, 0)
vbox.reorder_child(toolbar, -1)

def update(event):
    if False:
        while True:
            i = 10
    if event.xdata is None:
        label.set_markup('Drag mouse over axes for position')
    else:
        label.set_markup(f'<span color="#ef0000">x,y=({event.xdata}, {event.ydata})</span>')
fig.canvas.mpl_connect('motion_notify_event', update)
plt.show()