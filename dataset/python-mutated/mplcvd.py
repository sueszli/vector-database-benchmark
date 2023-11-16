"""
mplcvd -- an example of figure hook
===================================

To use this hook, ensure that this module is in your ``PYTHONPATH``, and set
``rcParams["figure.hooks"] = ["mplcvd:setup"]``.  This hook depends on
the ``colorspacious`` third-party module.
"""
import functools
from pathlib import Path
import colorspacious
import numpy as np
_BUTTON_NAME = 'Filter'
_BUTTON_HELP = 'Simulate color vision deficiencies'
_MENU_ENTRIES = {'None': None, 'Greyscale': 'greyscale', 'Deuteranopia': 'deuteranomaly', 'Protanopia': 'protanomaly', 'Tritanopia': 'tritanomaly'}

def _get_color_filter(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a color filter name, create a color filter function.\n\n    Parameters\n    ----------\n    name : str\n        The color filter name, one of the following:\n\n        - ``"none"``: ...\n        - ``"greyscale"``: Convert the input to luminosity.\n        - ``"deuteranopia"``: Simulate the most common form of red-green\n          colorblindness.\n        - ``"protanopia"``: Simulate a rarer form of red-green colorblindness.\n        - ``"tritanopia"``: Simulate the rare form of blue-yellow\n          colorblindness.\n\n        Color conversions use `colorspacious`_.\n\n    Returns\n    -------\n    callable\n        A color filter function that has the form:\n\n        def filter(input: np.ndarray[M, N, D])-> np.ndarray[M, N, D]\n\n        where (M, N) are the image dimensions, and D is the color depth (3 for\n        RGB, 4 for RGBA). Alpha is passed through unchanged and otherwise\n        ignored.\n    '
    if name not in _MENU_ENTRIES:
        raise ValueError(f'Unsupported filter name: {name!r}')
    name = _MENU_ENTRIES[name]
    if name is None:
        return None
    elif name == 'greyscale':
        rgb_to_jch = colorspacious.cspace_converter('sRGB1', 'JCh')
        jch_to_rgb = colorspacious.cspace_converter('JCh', 'sRGB1')

        def convert(im):
            if False:
                for i in range(10):
                    print('nop')
            greyscale_JCh = rgb_to_jch(im)
            greyscale_JCh[..., 1] = 0
            im = jch_to_rgb(greyscale_JCh)
            return im
    else:
        cvd_space = {'name': 'sRGB1+CVD', 'cvd_type': name, 'severity': 100}
        convert = colorspacious.cspace_converter(cvd_space, 'sRGB1')

    def filter_func(im, dpi):
        if False:
            for i in range(10):
                print('nop')
        alpha = None
        if im.shape[-1] == 4:
            (im, alpha) = (im[..., :3], im[..., 3])
        im = convert(im)
        if alpha is not None:
            im = np.dstack((im, alpha))
        return (np.clip(im, 0, 1), 0, 0)
    return filter_func

def _set_menu_entry(tb, name):
    if False:
        return 10
    tb.canvas.figure.set_agg_filter(_get_color_filter(name))
    tb.canvas.draw_idle()

def setup(figure):
    if False:
        for i in range(10):
            print('nop')
    tb = figure.canvas.toolbar
    if tb is None:
        return
    for cls in type(tb).__mro__:
        pkg = cls.__module__.split('.')[0]
        if pkg != 'matplotlib':
            break
    if pkg == 'gi':
        _setup_gtk(tb)
    elif pkg in ('PyQt5', 'PySide2', 'PyQt6', 'PySide6'):
        _setup_qt(tb)
    elif pkg == 'tkinter':
        _setup_tk(tb)
    elif pkg == 'wx':
        _setup_wx(tb)
    else:
        raise NotImplementedError('The current backend is not supported')

def _setup_gtk(tb):
    if False:
        i = 10
        return i + 15
    from gi.repository import Gio, GLib, Gtk
    for idx in range(tb.get_n_items()):
        children = tb.get_nth_item(idx).get_children()
        if children and isinstance(children[0], Gtk.Label):
            break
    toolitem = Gtk.SeparatorToolItem()
    tb.insert(toolitem, idx)
    image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(str(Path(__file__).parent / 'images/eye-symbolic.svg')), Gtk.IconSize.LARGE_TOOLBAR)
    if Gtk.check_version(3, 6, 0) is None:
        group = Gio.SimpleActionGroup.new()
        action = Gio.SimpleAction.new_stateful('cvdsim', GLib.VariantType('s'), GLib.Variant('s', 'none'))
        group.add_action(action)

        @functools.partial(action.connect, 'activate')
        def set_filter(action, parameter):
            if False:
                print('Hello World!')
            _set_menu_entry(tb, parameter.get_string())
            action.set_state(parameter)
        menu = Gio.Menu()
        for name in _MENU_ENTRIES:
            menu.append(name, f'local.cvdsim::{name}')
        button = Gtk.MenuButton.new()
        button.remove(button.get_children()[0])
        button.add(image)
        button.insert_action_group('local', group)
        button.set_menu_model(menu)
        button.get_style_context().add_class('flat')
        item = Gtk.ToolItem()
        item.add(button)
        tb.insert(item, idx + 1)
    else:
        menu = Gtk.Menu()
        group = []
        for name in _MENU_ENTRIES:
            item = Gtk.RadioMenuItem.new_with_label(group, name)
            item.set_active(name == 'None')
            item.connect('activate', lambda item: _set_menu_entry(tb, item.get_label()))
            group.append(item)
            menu.append(item)
        menu.show_all()
        tbutton = Gtk.MenuToolButton.new(image, _BUTTON_NAME)
        tbutton.set_menu(menu)
        tb.insert(tbutton, idx + 1)
    tb.show_all()

def _setup_qt(tb):
    if False:
        print('Hello World!')
    from matplotlib.backends.qt_compat import QtGui, QtWidgets
    menu = QtWidgets.QMenu()
    try:
        QActionGroup = QtGui.QActionGroup
    except AttributeError:
        QActionGroup = QtWidgets.QActionGroup
    group = QActionGroup(menu)
    group.triggered.connect(lambda action: _set_menu_entry(tb, action.text()))
    for name in _MENU_ENTRIES:
        action = menu.addAction(name)
        action.setCheckable(True)
        action.setActionGroup(group)
        action.setChecked(name == 'None')
    actions = tb.actions()
    before = next((action for action in actions if isinstance(tb.widgetForAction(action), QtWidgets.QLabel)), None)
    tb.insertSeparator(before)
    button = QtWidgets.QToolButton()
    button.setIcon(tb._icon(str(Path(__file__).parent / 'images/eye.png')))
    button.setText(_BUTTON_NAME)
    button.setToolTip(_BUTTON_HELP)
    button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setMenu(menu)
    tb.insertWidget(before, button)

def _setup_tk(tb):
    if False:
        print('Hello World!')
    import tkinter as tk
    tb._Spacer()
    button = tk.Menubutton(master=tb, relief='raised')
    button._image_file = str(Path(__file__).parent / 'images/eye.png')
    tb._set_image_for_button(button)
    button.pack(side=tk.LEFT)
    menu = tk.Menu(master=button, tearoff=False)
    for name in _MENU_ENTRIES:
        menu.add('radiobutton', label=name, command=lambda _name=name: _set_menu_entry(tb, _name))
    menu.invoke(0)
    button.config(menu=menu)

def _setup_wx(tb):
    if False:
        i = 10
        return i + 15
    import wx
    idx = next((idx for idx in range(tb.ToolsCount) if tb.GetToolByPos(idx).IsStretchableSpace()))
    tb.InsertSeparator(idx)
    tool = tb.InsertTool(idx + 1, -1, _BUTTON_NAME, tb._icon(str(Path(__file__).parent / 'images/eye.png')), kind=wx.ITEM_DROPDOWN, shortHelp=_BUTTON_HELP)
    menu = wx.Menu()
    for name in _MENU_ENTRIES:
        item = menu.AppendRadioItem(-1, name)
        menu.Bind(wx.EVT_MENU, lambda event, _name=name: _set_menu_entry(tb, _name), id=item.Id)
    tb.SetDropdownMenu(tool.Id, menu)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cbook
    plt.rcParams['figure.hooks'].append('mplcvd:setup')
    (fig, axd) = plt.subplot_mosaic([['viridis', 'turbo'], ['photo', 'lines']])
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    (X, Y) = np.meshgrid(x, y)
    Z1 = np.exp(-X ** 2 - Y ** 2)
    Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2
    imv = axd['viridis'].imshow(Z, interpolation='bilinear', origin='lower', extent=[-3, 3, -3, 3], vmax=abs(Z).max(), vmin=-abs(Z).max())
    fig.colorbar(imv)
    imt = axd['turbo'].imshow(Z, interpolation='bilinear', cmap='turbo', origin='lower', extent=[-3, 3, -3, 3], vmax=abs(Z).max(), vmin=-abs(Z).max())
    fig.colorbar(imt)
    with cbook.get_sample_data('grace_hopper.jpg') as image_file:
        photo = plt.imread(image_file)
    axd['photo'].imshow(photo)
    th = np.linspace(0, 2 * np.pi, 1024)
    for j in [1, 2, 4, 6]:
        axd['lines'].plot(th, np.sin(th * j), label=f'$\\omega={j}$')
    axd['lines'].legend(ncols=2, loc='upper right')
    plt.show()