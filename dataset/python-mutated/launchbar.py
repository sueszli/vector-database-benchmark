"""
This module define a widget that displays icons to launch softwares or commands
when clicked -- a launchbar.
Only png icon files are displayed, not xpm because cairo doesn't support
loading of xpm file.
The order of displaying (from left to right) is in the order of the list.

If no icon was found for the name provided and if default_icon is set to None
then the name is printed instead. If default_icon is defined then this icon is
displayed instead.

To execute a software:
 - ('thunderbird', 'thunderbird -safe-mode', 'launch thunderbird in safe mode')
To execute a python command in qtile, begin with by 'qshell:'
 - ('/path/to/icon.png', 'qshell:self.qtile.shutdown()', 'logout from qtile')


"""
from __future__ import annotations
import os.path
import cairocffi
try:
    from xdg.IconTheme import getIconPath
    has_xdg = True
except ImportError:
    has_xdg = False
from libqtile import bar
from libqtile.images import Img
from libqtile.log_utils import logger
from libqtile.widget import base

class LaunchBar(base._Widget):
    """
    A widget that display icons to launch the associated command.

    Text will displayed when no icon is found.

    Optional requirements: `pyxdg <https://pypi.org/project/pyxdg/>`__ for finding the icon path if it is not provided in the ``progs`` tuple.
    """
    orientations = base.ORIENTATION_HORIZONTAL
    defaults = [('padding', 2, 'Padding between icons'), ('default_icon', '/usr/share/icons/oxygen/256x256/mimetypes/application-x-executable.png', 'Default icon not found'), ('font', 'sans', 'Text font'), ('fontsize', None, 'Font pixel size. Calculated if None.'), ('fontshadow', None, 'Font shadow color, default is None (no shadow)'), ('foreground', '#ffffff', 'Text colour.'), ('progs', [], "A list of tuples (software_name or icon_path, command_to_execute, comment), for example: [('thunderbird', 'thunderbird -safe-mode', 'launch thunderbird in safe mode'),  ('/path/to/icon.png', 'qshell:self.qtile.shutdown()', 'logout from qtile')]"), ('text_only', False, "Don't use any icons."), ('icon_size', None, 'Size of icons. ``None`` to fit to bar.'), ('padding_y', 0, 'Vertical adjustment for icons.'), ('theme_path', None, 'Path to icon theme to be used by pyxdg for icons. ``None`` will use default icon theme.')]

    def __init__(self, _progs: list[tuple[str, str, str]] | None=None, width=bar.CALCULATED, **config):
        if False:
            return 10
        base._Widget.__init__(self, width, **config)
        self.add_defaults(LaunchBar.defaults)
        self.surfaces: dict[str, Img | base._TextBox] = {}
        self.icons_files: dict[str, str | None] = {}
        self.icons_widths: dict[str, int] = {}
        self.icons_offsets: dict[str, int] = {}
        if _progs:
            logger.warning('The use of a positional argument in LaunchBar is deprecated. Please update your config to use progs=[...].')
            config['progs'] = _progs
        self.progs = dict(enumerate([{'name': prog[0], 'cmd': prog[1], 'comment': prog[2] if len(prog) > 2 else None} for prog in config.get('progs', list())]))
        self.progs_name = set([prog['name'] for prog in self.progs.values()])
        self.length_type = bar.STATIC
        self.length = 0

    def _configure(self, qtile, pbar):
        if False:
            return 10
        base._Widget._configure(self, qtile, pbar)
        self.lookup_icons()
        self.setup_images()
        self.length = self.calculate_length()

    def setup_images(self):
        if False:
            return 10
        'Create image structures for each icon files.'
        self._icon_size = self.icon_size if self.icon_size is not None else self.bar.height - 4
        self._icon_padding = (self.bar.height - self._icon_size) // 2
        for (img_name, iconfile) in self.icons_files.items():
            if iconfile is None or self.text_only:
                if not self.text_only:
                    logger.warning('No icon found for application "%s" (%s) switch to text mode', img_name, iconfile)
                textbox = base._TextBox()
                textbox._configure(self.qtile, self.bar)
                textbox.layout = self.drawer.textlayout(textbox.text, self.foreground, self.font, self.fontsize, self.fontshadow, markup=textbox.markup)
                textbox.text = img_name
                textbox.calculate_length()
                self.icons_widths[img_name] = textbox.width
                self.surfaces[img_name] = textbox
                continue
            else:
                try:
                    img = Img.from_path(iconfile)
                except cairocffi.Error:
                    logger.exception('Error loading icon for application "%s" (%s)', img_name, iconfile)
                    return
            input_width = img.width
            input_height = img.height
            sp = input_height / self._icon_size
            width = int(input_width / sp)
            imgpat = cairocffi.SurfacePattern(img.surface)
            scaler = cairocffi.Matrix()
            scaler.scale(sp, sp)
            scaler.translate(self.padding * -1, -2)
            imgpat.set_matrix(scaler)
            imgpat.set_filter(cairocffi.FILTER_BEST)
            self.surfaces[img_name] = imgpat
            self.icons_widths[img_name] = width

    def _lookup_icon(self, name):
        if False:
            print('Hello World!')
        'Search for the icon corresponding to one command.'
        self.icons_files[name] = None
        ipath = os.path.expanduser(name)
        if os.path.isabs(ipath):
            (root, ext) = os.path.splitext(ipath)
            img_extensions = ['.tif', '.tiff', '.bmp', '.jpg', '.jpeg', '.gif', '.png', '.svg']
            if ext in img_extensions:
                self.icons_files[name] = ipath if os.path.isfile(ipath) else None
            else:
                for extension in img_extensions:
                    if os.path.isfile(ipath + extension):
                        self.icons_files[name] = ipath + extension
                        break
        elif has_xdg:
            self.icons_files[name] = getIconPath(name, theme=self.theme_path)
        if self.icons_files[name] is None:
            self.icons_files[name] = self.default_icon

    def lookup_icons(self):
        if False:
            i = 10
            return i + 15
        'Search for the icons corresponding to the commands to execute.'
        if self.default_icon is not None:
            if not os.path.isfile(self.default_icon):
                self.default_icon = None
        for name in self.progs_name:
            self._lookup_icon(name)

    def get_icon_in_position(self, x, y):
        if False:
            i = 10
            return i + 15
        'Determine which icon is clicked according to its position.'
        for i in self.progs:
            if x < self.icons_offsets[i] + self.icons_widths[self.progs[i]['name']] + self.padding / 2:
                return i

    def button_press(self, x, y, button):
        if False:
            i = 10
            return i + 15
        'Launch the associated command to the clicked icon.'
        base._Widget.button_press(self, x, y, button)
        if button == 1:
            icon = self.get_icon_in_position(x, y)
            if icon is not None:
                cmd = self.progs[icon]['cmd']
                if cmd.startswith('qshell:'):
                    exec(cmd[7:].lstrip())
                else:
                    self.qtile.spawn(cmd)
            self.draw()

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        'Draw the icons in the widget.'
        self.drawer.clear(self.background or self.bar.background)
        xoffset = 0
        for i in sorted(self.progs.keys()):
            self.drawer.ctx.save()
            self.drawer.ctx.translate(xoffset, 0)
            self.icons_offsets[i] = xoffset + self.padding
            name = self.progs[i]['name']
            icon_width = self.icons_widths[name]
            if isinstance(self.surfaces[name], base._TextBox):
                textbox = self.surfaces[name]
                textbox.layout.draw(self.padding + textbox.actual_padding, int((self.bar.height - textbox.layout.height) / 2.0) + 1)
            else:
                self.drawer.ctx.save()
                self.drawer.ctx.translate(0, self._icon_padding + self.padding_y)
                self.drawer.ctx.set_source(self.surfaces[name])
                self.drawer.ctx.paint()
                self.drawer.ctx.restore()
            self.drawer.ctx.restore()
            self.drawer.draw(offsetx=self.offset + xoffset, offsety=self.offsety, width=icon_width + self.padding)
            xoffset += icon_width + self.padding
        self.drawer.draw(offsetx=self.offset, offsety=self.offsety, width=self.width)

    def calculate_length(self):
        if False:
            return 10
        'Compute the width of the widget according to each icon width.'
        return sum((self.icons_widths[prg['name']] for prg in self.progs.values())) + self.padding * (len(self.progs) + 1)