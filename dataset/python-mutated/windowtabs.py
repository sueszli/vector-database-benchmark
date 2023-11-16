from libqtile import bar, hook, pangocffi
from libqtile.log_utils import logger
from libqtile.widget import base

class WindowTabs(base._TextBox):
    """
    Displays the name of each window in the current group.
    Contrary to TaskList this is not an interactive widget.
    The window that currently has focus is highlighted.
    """
    defaults = [('separator', ' | ', 'Task separator text.'), ('selected', ('<b>', '</b>'), 'Selected task indicator'), ('parse_text', None, 'Function to parse and modify window names. e.g. function in config that removes excess strings from window name: def my_func(text)    for string in [" - Chromium", " - Firefox"]:        text = text.replace(string, "")   return textthen set option parse_text=my_func')]

    def __init__(self, **config):
        if False:
            print('Hello World!')
        width = config.pop('width', bar.STRETCH)
        base._TextBox.__init__(self, width=width, **config)
        self.add_defaults(WindowTabs.defaults)
        if not isinstance(self.selected, (tuple, list)):
            self.selected = (self.selected, self.selected)

    def _configure(self, qtile, bar):
        if False:
            for i in range(10):
                print('nop')
        base._TextBox._configure(self, qtile, bar)
        hook.subscribe.client_name_updated(self.update)
        hook.subscribe.focus_change(self.update)
        hook.subscribe.float_change(self.update)
        self.add_callbacks({'Button1': self.bar.screen.group.next_window})

    def update(self, *args):
        if False:
            i = 10
            return i + 15
        names = []
        for w in self.bar.screen.group.windows:
            state = ''
            if w.maximized:
                state = '[] '
            elif w.minimized:
                state = '_ '
            elif w.floating:
                state = 'V '
            task = '%s%s' % (state, w.name if w and w.name else ' ')
            task = pangocffi.markup_escape_text(task)
            if w is self.bar.screen.group.current_window:
                task = task.join(self.selected)
            names.append(task)
        self.text = self.separator.join(names)
        if callable(self.parse_text):
            try:
                self.text = self.parse_text(self.text)
            except:
                logger.exception('parse_text function failed:')
        self.bar.draw()