"""
A display-only column that displays any data type.
"""
from typing import Any
import urwid
from mitmproxy.tools.console.grideditor import base
from mitmproxy.utils import strutils

class Column(base.Column):

    def Display(self, data):
        if False:
            i = 10
            return i + 15
        return Display(data)
    Edit = Display

    def blank(self):
        if False:
            print('Hello World!')
        return ''

class Display(base.Cell):

    def __init__(self, data: Any) -> None:
        if False:
            return 10
        self.data = data
        if isinstance(data, bytes):
            data = strutils.bytes_to_escaped_str(data)
        if not isinstance(data, str):
            data = repr(data)
        w = urwid.Text(data, wrap='any')
        super().__init__(w)

    def get_data(self) -> Any:
        if False:
            print('Hello World!')
        return self.data