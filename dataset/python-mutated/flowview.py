import logging
import math
import sys
from functools import lru_cache
import urwid
import mitmproxy.flow
import mitmproxy.tools.console.master
from mitmproxy import contentviews
from mitmproxy import ctx
from mitmproxy import dns
from mitmproxy import http
from mitmproxy import tcp
from mitmproxy import udp
from mitmproxy.tools.console import common
from mitmproxy.tools.console import flowdetailview
from mitmproxy.tools.console import layoutwidget
from mitmproxy.tools.console import searchable
from mitmproxy.tools.console import tabs
from mitmproxy.utils import strutils

class SearchError(Exception):
    pass

class FlowViewHeader(urwid.WidgetWrap):

    def __init__(self, master: 'mitmproxy.tools.console.master.ConsoleMaster') -> None:
        if False:
            i = 10
            return i + 15
        self.master = master
        self.focus_changed()

    def focus_changed(self):
        if False:
            i = 10
            return i + 15
        (cols, _) = self.master.ui.get_cols_rows()
        if self.master.view.focus.flow:
            self._w = common.format_flow(self.master.view.focus.flow, render_mode=common.RenderMode.DETAILVIEW, hostheader=self.master.options.showhost)
        else:
            self._w = urwid.Pile([])

class FlowDetails(tabs.Tabs):

    def __init__(self, master):
        if False:
            for i in range(10):
                print('nop')
        self.master = master
        super().__init__([])
        self.show()
        self.last_displayed_body = None
        contentviews.on_add.connect(self.contentview_changed)
        contentviews.on_remove.connect(self.contentview_changed)

    @property
    def view(self):
        if False:
            i = 10
            return i + 15
        return self.master.view

    @property
    def flow(self) -> mitmproxy.flow.Flow:
        if False:
            return 10
        return self.master.view.focus.flow

    def contentview_changed(self, view):
        if False:
            return 10
        self._get_content_view.cache_clear()
        if self.master.window.current_window('flowview'):
            self.show()

    def focus_changed(self):
        if False:
            print('Hello World!')
        f = self.flow
        if f:
            if isinstance(f, http.HTTPFlow):
                if f.websocket:
                    self.tabs = [(self.tab_http_request, self.view_request), (self.tab_http_response, self.view_response), (self.tab_websocket_messages, self.view_websocket_messages), (self.tab_details, self.view_details)]
                else:
                    self.tabs = [(self.tab_http_request, self.view_request), (self.tab_http_response, self.view_response), (self.tab_details, self.view_details)]
            elif isinstance(f, tcp.TCPFlow):
                self.tabs = [(self.tab_tcp_stream, self.view_message_stream), (self.tab_details, self.view_details)]
            elif isinstance(f, udp.UDPFlow):
                self.tabs = [(self.tab_udp_stream, self.view_message_stream), (self.tab_details, self.view_details)]
            elif isinstance(f, dns.DNSFlow):
                self.tabs = [(self.tab_dns_request, self.view_dns_request), (self.tab_dns_response, self.view_dns_response), (self.tab_details, self.view_details)]
            self.show()
        else:
            self.master.window.pop()

    def tab_http_request(self):
        if False:
            for i in range(10):
                print('nop')
        flow = self.flow
        assert isinstance(flow, http.HTTPFlow)
        if self.flow.intercepted and (not flow.response):
            return 'Request intercepted'
        else:
            return 'Request'

    def tab_http_response(self):
        if False:
            return 10
        flow = self.flow
        assert isinstance(flow, http.HTTPFlow)
        if self.flow.intercepted and flow.response:
            return 'Response intercepted'
        else:
            return 'Response'

    def tab_dns_request(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        flow = self.flow
        assert isinstance(flow, dns.DNSFlow)
        if self.flow.intercepted and (not flow.response):
            return 'Request intercepted'
        else:
            return 'Request'

    def tab_dns_response(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        flow = self.flow
        assert isinstance(flow, dns.DNSFlow)
        if self.flow.intercepted and flow.response:
            return 'Response intercepted'
        else:
            return 'Response'

    def tab_tcp_stream(self):
        if False:
            return 10
        return 'TCP Stream'

    def tab_udp_stream(self):
        if False:
            while True:
                i = 10
        return 'UDP Stream'

    def tab_websocket_messages(self):
        if False:
            while True:
                i = 10
        return 'WebSocket Messages'

    def tab_details(self):
        if False:
            while True:
                i = 10
        return 'Detail'

    def view_request(self):
        if False:
            while True:
                i = 10
        flow = self.flow
        assert isinstance(flow, http.HTTPFlow)
        return self.conn_text(flow.request)

    def view_response(self):
        if False:
            return 10
        flow = self.flow
        assert isinstance(flow, http.HTTPFlow)
        return self.conn_text(flow.response)

    def view_dns_request(self):
        if False:
            i = 10
            return i + 15
        flow = self.flow
        assert isinstance(flow, dns.DNSFlow)
        return self.dns_message_text('request', flow.request)

    def view_dns_response(self):
        if False:
            while True:
                i = 10
        flow = self.flow
        assert isinstance(flow, dns.DNSFlow)
        return self.dns_message_text('response', flow.response)

    def _contentview_status_bar(self, description: str, viewmode: str):
        if False:
            while True:
                i = 10
        cols = [urwid.Text([('heading', description)]), urwid.Text([' ', ('heading', '['), ('heading_key', 'm'), ('heading', ':%s]' % viewmode)], align='right')]
        contentview_status_bar = urwid.AttrWrap(urwid.Columns(cols), 'heading')
        return contentview_status_bar
    FROM_CLIENT_MARKER = ('from_client', f'{common.SYMBOL_FROM_CLIENT} ')
    TO_CLIENT_MARKER = ('to_client', f'{common.SYMBOL_TO_CLIENT} ')

    def view_websocket_messages(self):
        if False:
            return 10
        flow = self.flow
        assert isinstance(flow, http.HTTPFlow)
        assert flow.websocket is not None
        if not flow.websocket.messages:
            return searchable.Searchable([urwid.Text(('highlight', 'No messages.'))])
        viewmode = self.master.commands.call('console.flowview.mode')
        widget_lines = []
        for m in flow.websocket.messages:
            (_, lines, _) = contentviews.get_message_content_view(viewmode, m, flow)
            for line in lines:
                if m.from_client:
                    line.insert(0, self.FROM_CLIENT_MARKER)
                else:
                    line.insert(0, self.TO_CLIENT_MARKER)
                widget_lines.append(urwid.Text(line))
        if flow.websocket.closed_by_client is not None:
            widget_lines.append(urwid.Text([self.FROM_CLIENT_MARKER if flow.websocket.closed_by_client else self.TO_CLIENT_MARKER, ('alert' if flow.websocket.close_code in (1000, 1001, 1005) else 'error', f'Connection closed: {flow.websocket.close_code} {flow.websocket.close_reason}')]))
        if flow.intercepted:
            markup = widget_lines[-1].get_text()[0]
            widget_lines[-1].set_text(('intercept', markup))
        widget_lines.insert(0, self._contentview_status_bar(viewmode.capitalize(), viewmode))
        return searchable.Searchable(widget_lines)

    def view_message_stream(self) -> urwid.Widget:
        if False:
            print('Hello World!')
        flow = self.flow
        assert isinstance(flow, (tcp.TCPFlow, udp.UDPFlow))
        if not flow.messages:
            return searchable.Searchable([urwid.Text(('highlight', 'No messages.'))])
        viewmode = self.master.commands.call('console.flowview.mode')
        widget_lines = []
        for m in flow.messages:
            (_, lines, _) = contentviews.get_message_content_view(viewmode, m, flow)
            for line in lines:
                if m.from_client:
                    line.insert(0, self.FROM_CLIENT_MARKER)
                else:
                    line.insert(0, self.TO_CLIENT_MARKER)
                widget_lines.append(urwid.Text(line))
        if flow.intercepted:
            markup = widget_lines[-1].get_text()[0]
            widget_lines[-1].set_text(('intercept', markup))
        widget_lines.insert(0, self._contentview_status_bar(viewmode.capitalize(), viewmode))
        return searchable.Searchable(widget_lines)

    def view_details(self):
        if False:
            for i in range(10):
                print('nop')
        return flowdetailview.flowdetails(self.view, self.flow)

    def content_view(self, viewmode, message):
        if False:
            return 10
        if message.raw_content is None:
            (msg, body) = ('', [urwid.Text([('error', '[content missing]')])])
            return (msg, body)
        else:
            full = self.master.commands.execute('view.settings.getval @focus fullcontents false')
            if full == 'true':
                limit = sys.maxsize
            else:
                limit = ctx.options.content_view_lines_cutoff
            flow_modify_cache_invalidation = hash((message.raw_content, message.headers.fields, getattr(message, 'path', None)))
            self._get_content_view_message = message
            return self._get_content_view(viewmode, limit, flow_modify_cache_invalidation)

    @lru_cache(maxsize=200)
    def _get_content_view(self, viewmode, max_lines, _):
        if False:
            print('Hello World!')
        message = self._get_content_view_message
        self._get_content_view_message = None
        (description, lines, error) = contentviews.get_message_content_view(viewmode, message, self.flow)
        if error:
            logging.debug(error)
        if description == 'No content' and isinstance(message, http.Request):
            description = 'No request content'
        chars_per_line = 80
        max_chars = max_lines * chars_per_line
        total_chars = 0
        text_objects = []
        for line in lines:
            txt = []
            for (style, text) in line:
                if total_chars + len(text) > max_chars:
                    text = text[:max_chars - total_chars]
                txt.append((style, text))
                total_chars += len(text)
                if total_chars == max_chars:
                    break
            total_chars = int(math.ceil(total_chars / chars_per_line) * chars_per_line)
            text_objects.append(urwid.Text(txt))
            if total_chars == max_chars:
                text_objects.append(urwid.Text([('highlight', 'Stopped displaying data after %d lines. Press ' % max_lines), ('key', 'f'), ('highlight', ' to load all data.')]))
                break
        return (description, text_objects)

    def conn_text(self, conn):
        if False:
            for i in range(10):
                print('nop')
        if conn:
            hdrs = []
            for (k, v) in conn.headers.fields:
                k = strutils.bytes_to_escaped_str(k) + ':'
                v = strutils.bytes_to_escaped_str(v)
                hdrs.append((k, v))
            txt = common.format_keyvals(hdrs, key_format='header')
            viewmode = self.master.commands.call('console.flowview.mode')
            (msg, body) = self.content_view(viewmode, conn)
            cols = [urwid.Text([('heading', msg)]), urwid.Text([' ', ('heading', '['), ('heading_key', 'm'), ('heading', ':%s]' % viewmode)], align='right')]
            title = urwid.AttrWrap(urwid.Columns(cols), 'heading')
            txt.append(title)
            txt.extend(body)
        else:
            txt = [urwid.Text(''), urwid.Text([('highlight', 'No response. Press '), ('key', 'e'), ('highlight', ' and edit any aspect to add one.')])]
        return searchable.Searchable(txt)

    def dns_message_text(self, type: str, message: dns.Message | None) -> searchable.Searchable:
        if False:
            print('Hello World!')
        if message:

            def rr_text(rr: dns.ResourceRecord):
                if False:
                    while True:
                        i = 10
                return urwid.Text(f'  {rr.name} {dns.types.to_str(rr.type)} {dns.classes.to_str(rr.class_)} {rr.ttl} {str(rr)}')
            txt = []
            txt.append(urwid.Text('{recursive}Question'.format(recursive='Recursive ' if message.recursion_desired else '')))
            txt.extend((urwid.Text(f'  {q.name} {dns.types.to_str(q.type)} {dns.classes.to_str(q.class_)}') for q in message.questions))
            txt.append(urwid.Text(''))
            txt.append(urwid.Text('{authoritative}{recursive}Answer'.format(authoritative='Authoritative ' if message.authoritative_answer else '', recursive='Recursive ' if message.recursion_available else '')))
            txt.extend(map(rr_text, message.answers))
            txt.append(urwid.Text(''))
            txt.append(urwid.Text('Authority'))
            txt.extend(map(rr_text, message.authorities))
            txt.append(urwid.Text(''))
            txt.append(urwid.Text('Addition'))
            txt.extend(map(rr_text, message.additionals))
            return searchable.Searchable(txt)
        else:
            return searchable.Searchable([urwid.Text(('highlight', f'No {type}.'))])

class FlowView(urwid.Frame, layoutwidget.LayoutWidget):
    keyctx = 'flowview'
    title = 'Flow Details'

    def __init__(self, master):
        if False:
            print('Hello World!')
        super().__init__(FlowDetails(master), header=FlowViewHeader(master))
        self.master = master

    def focus_changed(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.body.focus_changed()
        self.header.focus_changed()