from typing import Any
import urwid
from mitmproxy import exceptions
from mitmproxy.http import Headers
from mitmproxy.tools.console import layoutwidget
from mitmproxy.tools.console import signals
from mitmproxy.tools.console.grideditor import base
from mitmproxy.tools.console.grideditor import col_bytes
from mitmproxy.tools.console.grideditor import col_subgrid
from mitmproxy.tools.console.grideditor import col_text
from mitmproxy.tools.console.grideditor import col_viewany

class QueryEditor(base.FocusEditor):
    title = 'Edit Query'
    columns = [col_text.Column('Key'), col_text.Column('Value')]

    def get_data(self, flow):
        if False:
            return 10
        return flow.request.query.items(multi=True)

    def set_data(self, vals, flow):
        if False:
            i = 10
            return i + 15
        flow.request.query = vals

class HeaderEditor(base.FocusEditor):
    columns = [col_bytes.Column('Key'), col_bytes.Column('Value')]

class RequestHeaderEditor(HeaderEditor):
    title = 'Edit Request Headers'

    def get_data(self, flow):
        if False:
            return 10
        return flow.request.headers.fields

    def set_data(self, vals, flow):
        if False:
            while True:
                i = 10
        flow.request.headers = Headers(vals)

class ResponseHeaderEditor(HeaderEditor):
    title = 'Edit Response Headers'

    def get_data(self, flow):
        if False:
            for i in range(10):
                print('nop')
        return flow.response.headers.fields

    def set_data(self, vals, flow):
        if False:
            i = 10
            return i + 15
        flow.response.headers = Headers(vals)

class RequestMultipartEditor(base.FocusEditor):
    title = 'Edit Multipart Form'
    columns = [col_bytes.Column('Key'), col_bytes.Column('Value')]

    def get_data(self, flow):
        if False:
            i = 10
            return i + 15
        return flow.request.multipart_form.items(multi=True)

    def set_data(self, vals, flow):
        if False:
            print('Hello World!')
        flow.request.multipart_form = vals

class RequestUrlEncodedEditor(base.FocusEditor):
    title = 'Edit UrlEncoded Form'
    columns = [col_text.Column('Key'), col_text.Column('Value')]

    def get_data(self, flow):
        if False:
            print('Hello World!')
        return flow.request.urlencoded_form.items(multi=True)

    def set_data(self, vals, flow):
        if False:
            return 10
        flow.request.urlencoded_form = vals

class PathEditor(base.FocusEditor):
    title = 'Edit Path Components'
    columns = [col_text.Column('Component')]

    def data_in(self, data):
        if False:
            print('Hello World!')
        return [[i] for i in data]

    def data_out(self, data):
        if False:
            i = 10
            return i + 15
        return [i[0] for i in data]

    def get_data(self, flow):
        if False:
            while True:
                i = 10
        return self.data_in(flow.request.path_components)

    def set_data(self, vals, flow):
        if False:
            while True:
                i = 10
        flow.request.path_components = self.data_out(vals)

class CookieEditor(base.FocusEditor):
    title = 'Edit Cookies'
    columns = [col_text.Column('Name'), col_text.Column('Value')]

    def get_data(self, flow):
        if False:
            print('Hello World!')
        return flow.request.cookies.items(multi=True)

    def set_data(self, vals, flow):
        if False:
            return 10
        flow.request.cookies = vals

class CookieAttributeEditor(base.FocusEditor):
    title = 'Editing Set-Cookie attributes'
    columns = [col_text.Column('Name'), col_text.Column('Value')]
    grideditor: base.BaseGridEditor

    def data_in(self, data):
        if False:
            return 10
        return [(k, v or '') for (k, v) in data]

    def data_out(self, data):
        if False:
            i = 10
            return i + 15
        ret = []
        for i in data:
            if not i[1]:
                ret.append([i[0], None])
            else:
                ret.append(i)
        return ret

    def layout_pushed(self, prev):
        if False:
            print('Hello World!')
        if self.grideditor.master.view.focus.flow:
            self._w = base.BaseGridEditor(self.grideditor.master, self.title, self.columns, self.grideditor.walker.get_current_value(), self.grideditor.set_subeditor_value, self.grideditor.walker.focus, self.grideditor.walker.focus_col)
        else:
            self._w = urwid.Pile([])

class SetCookieEditor(base.FocusEditor):
    title = 'Edit SetCookie Header'
    columns = [col_text.Column('Name'), col_text.Column('Value'), col_subgrid.Column('Attributes', CookieAttributeEditor)]

    def data_in(self, data):
        if False:
            while True:
                i = 10
        flattened = []
        for (key, (value, attrs)) in data:
            flattened.append([key, value, attrs.items(multi=True)])
        return flattened

    def data_out(self, data):
        if False:
            while True:
                i = 10
        vals = []
        for (key, value, attrs) in data:
            vals.append([key, (value, attrs)])
        return vals

    def get_data(self, flow):
        if False:
            for i in range(10):
                print('nop')
        return self.data_in(flow.response.cookies.items(multi=True))

    def set_data(self, vals, flow):
        if False:
            return 10
        flow.response.cookies = self.data_out(vals)

class OptionsEditor(base.GridEditor, layoutwidget.LayoutWidget):
    title = ''
    columns = [col_text.Column('')]

    def __init__(self, master, name, vals):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        super().__init__(master, [[i] for i in vals], self.callback)

    def callback(self, vals) -> None:
        if False:
            i = 10
            return i + 15
        try:
            setattr(self.master.options, self.name, [i[0] for i in vals])
        except exceptions.OptionsError as v:
            signals.status_message.send(message=str(v))

    def is_error(self, col, val):
        if False:
            while True:
                i = 10
        pass

class DataViewer(base.GridEditor, layoutwidget.LayoutWidget):
    title = ''

    def __init__(self, master, vals: list[list[Any]] | list[Any] | Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if vals is not None:
            if not isinstance(vals, list):
                vals = [vals]
            if not isinstance(vals[0], list):
                vals = [[i] for i in vals]
            self.columns = [col_viewany.Column('')] * len(vals[0])
        super().__init__(master, vals, self.callback)

    def callback(self, vals):
        if False:
            print('Hello World!')
        pass

    def is_error(self, col, val):
        if False:
            while True:
                i = 10
        pass