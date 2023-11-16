import logging
import gi
from gi.repository import Gtk
logger = logging.getLogger()
try:
    gi.require_version('GtkLayerShell', '0.1')
    from gi.repository import GtkLayerShell
except (ValueError, ImportError):
    GtkLayerShell = None

class LayerShellOverlay(Gtk.Window):
    """
    Allows for a window to opt in to the wayland layer shell protocol.

    This Disables decorations and displays the window on top of other applications (even if fullscreen).
    Uses the wlr-layer-shell protocol [1]

    [1]: https://gitlab.freedesktop.org/wlroots/wlr-protocols/-/blob/master/unstable/wlr-layer-shell-unstable-v1.xml
    """

    @classmethod
    def is_supported(cls):
        if False:
            return 10
        'Check if running under a wayland compositor that supports the layer shell extension'
        return GtkLayerShell is not None and GtkLayerShell.is_supported()

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._use_layer_shell = False

    def enable_layer_shell(self):
        if False:
            for i in range(10):
                print('nop')
        assert __class__.is_supported(), 'Should be supported to enable'
        self._use_layer_shell = True
        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.EXCLUSIVE)
        GtkLayerShell.set_layer(self, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_exclusive_zone(self, 0)

    @property
    def layer_shell_enabled(self):
        if False:
            i = 10
            return i + 15
        return self._use_layer_shell

    def set_vertical_position(self, pos_y):
        if False:
            print('Hello World!')
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.TOP, pos_y)