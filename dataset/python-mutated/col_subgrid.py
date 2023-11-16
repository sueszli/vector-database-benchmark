import urwid
from mitmproxy.net.http import cookies
from mitmproxy.tools.console import signals
from mitmproxy.tools.console.grideditor import base

class Column(base.Column):

    def __init__(self, heading, subeditor):
        if False:
            i = 10
            return i + 15
        super().__init__(heading)
        self.subeditor = subeditor

    def Edit(self, data):
        if False:
            return 10
        raise RuntimeError('SubgridColumn should handle edits itself')

    def Display(self, data):
        if False:
            i = 10
            return i + 15
        return Display(data)

    def blank(self):
        if False:
            for i in range(10):
                print('nop')
        return []

    def keypress(self, key: str, editor):
        if False:
            for i in range(10):
                print('nop')
        if key in 'rRe':
            signals.status_message.send(message='Press enter to edit this field.')
            return
        elif key == 'm_select':
            self.subeditor.grideditor = editor
            editor.master.switch_view('edit_focus_setcookie_attrs')
        else:
            return key

class Display(base.Cell):

    def __init__(self, data):
        if False:
            print('Hello World!')
        p = cookies._format_pairs(data, sep='\n')
        w = urwid.Text(p)
        super().__init__(w)

    def get_data(self):
        if False:
            for i in range(10):
                print('nop')
        pass