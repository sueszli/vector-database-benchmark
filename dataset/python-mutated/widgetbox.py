from __future__ import annotations
from typing import TYPE_CHECKING
from libqtile.command.base import expose_command
from libqtile.log_utils import logger
from libqtile.pangocffi import markup_escape_text
from libqtile.widget import Systray, base
if TYPE_CHECKING:
    from typing import Any

class WidgetBox(base._TextBox):
    """A widget to declutter your bar.

    WidgetBox is a widget that hides widgets by default but shows them when
    the box is opened.

    Widgets that are hidden will still update etc. as if they were on the main
    bar.

    Button clicks are passed to widgets when they are visible so callbacks will
    work.

    Widgets in the box also remain accessible via command interfaces.

    Widgets can only be added to the box via the configuration file. The widget
    is configured by adding widgets to the "widgets" parameter as follows::

        widget.WidgetBox(widgets=[
            widget.TextBox(text="This widget is in the box"),
            widget.Memory()
            ]
        ),
    """
    orientations = base.ORIENTATION_HORIZONTAL
    defaults: list[tuple[str, Any, str]] = [('close_button_location', 'left', "Location of close button when box open ('left' or 'right')"), ('text_closed', '[<]', 'Text when box is closed'), ('text_open', '[>]', 'Text when box is open'), ('widgets', list(), 'A list of widgets to include in the box'), ('start_opened', False, 'Spawn the box opened')]

    def __init__(self, _widgets: list[base._Widget] | None=None, **config):
        if False:
            i = 10
            return i + 15
        base._TextBox.__init__(self, **config)
        self.add_defaults(WidgetBox.defaults)
        self.box_is_open = False
        self.add_callbacks({'Button1': self.toggle})
        if _widgets:
            logger.warning('The use of a positional argument in WidgetBox is deprecated. Please update your config to use widgets=[...].')
            self.widgets = _widgets
        self.close_button_location: str
        if self.close_button_location not in ['left', 'right']:
            val = self.close_button_location
            logger.warning("Invalid value for 'close_button_location': %s", val)
            self.close_button_location = 'left'

    def _configure(self, qtile, bar):
        if False:
            return 10
        base._TextBox._configure(self, qtile, bar)
        self.text = markup_escape_text(self.text_open if self.box_is_open else self.text_closed)
        if self.configured:
            return
        for (idx, w) in enumerate(self.widgets):
            if w.configured:
                w = w.create_mirror()
                self.widgets[idx] = w
            self.qtile.register_widget(w)
            w._configure(self.qtile, self.bar)
            w.offsety = self.bar.border_width[0]
            w.offsetx = self.bar.width
            self.qtile.call_soon(w.draw)
            w.configured = True
        for w in self.widgets:
            w.drawer.disable()
        if self.start_opened and (not self.box_is_open):
            self.qtile.call_soon(self.toggle)

    def set_box_label(self):
        if False:
            i = 10
            return i + 15
        self.text = markup_escape_text(self.text_open if self.box_is_open else self.text_closed)

    def toggle_widgets(self):
        if False:
            while True:
                i = 10
        for widget in self.widgets:
            try:
                self.bar.widgets.remove(widget)
                widget.drawer.disable()
                if isinstance(widget, Systray):
                    for icon in widget.tray_icons:
                        icon.hide()
            except ValueError:
                continue
        index = self.bar.widgets.index(self)
        if self.close_button_location == 'left':
            index += 1
        if self.box_is_open:
            for widget in self.widgets[::-1]:
                widget.drawer.enable()
                self.bar.widgets.insert(index, widget)

    @expose_command()
    def toggle(self):
        if False:
            i = 10
            return i + 15
        'Toggle box state'
        self.box_is_open = not self.box_is_open
        self.toggle_widgets()
        self.set_box_label()
        self.bar.draw()