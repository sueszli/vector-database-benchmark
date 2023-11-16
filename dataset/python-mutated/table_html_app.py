import os
import json
import clastic
from clastic import Application
from clastic.render import JSONRender
from clastic.middleware import GetParamMiddleware
from clastic import Response
from clastic.sinter import getargspec
from boltons.tableutils import Table
_DATA = json.load(open('meta_stats.json'))
_CUR_PATH = os.path.dirname(os.path.abspath(clastic.__file__))
_CA_PATH = _CUR_PATH + '/_clastic_assets'
_CSS_PATH = _CA_PATH + '/common.css'
_STYLE = open(_CSS_PATH).read()
try:
    basestring
    unicode
except NameError:
    basestring = (str,)
    unicode = str

def fetch_json(url):
    if False:
        print('Hello World!')
    import urllib2
    response = urllib2.urlopen(url)
    content = response.read()
    data = json.loads(content)
    return data

class AutoTableRenderer(object):
    _html_doctype = '<!doctype html>'
    (_html_wrapper, _html_wrapper_close) = ('<html>', '</html>')
    _html_table_tag = '<table class="clastic-atr-table">'
    _html_style_content = _STYLE

    def __init__(self, max_depth=4, orientation='auto'):
        if False:
            while True:
                i = 10
        self.max_depth = max_depth
        self.orientation = orientation

    def _html_format_ep(self, route):
        if False:
            for i in range(10):
                print('nop')
        module_name = route.endpoint.__module__
        try:
            func_name = route.endpoint.func_name
        except:
            func_name = repr(route.endpoint)
        (args, _, _, _) = getargspec(route.endpoint)
        argstr = ', '.join(args)
        title = '<h2><small><sub>%s</sub></small><br/>%s(%s)</h2>' % (module_name, func_name, argstr)
        return title

    def __call__(self, context, _route):
        if False:
            for i in range(10):
                print('nop')
        content_parts = [self._html_wrapper]
        if self._html_style_content:
            content_parts.extend(['<head><style type="text/css">', self._html_style_content, '</style></head>'])
        content_parts.append('<body>')
        title = self._html_format_ep(_route)
        content_parts.append(title)
        table = Table.from_data(context, max_depth=self.max_depth)
        table._html_table_tag = self._html_table_tag
        content = table.to_html(max_depth=self.max_depth, orientation=self.orientation)
        content_parts.append(content)
        content_parts.append('</body>')
        content_parts.append(self._html_wrapper_close)
        return Response('\n'.join(content_parts), mimetype='text/html')

class BasicRender(object):
    _default_mime = 'application/json'
    _format_mime_map = {'html': 'text/html', 'json': 'application/json'}

    def __init__(self, dev_mode=True, qp_name='format'):
        if False:
            print('Hello World!')
        self.qp_name = qp_name
        self.json_render = JSONRender(dev_mode=dev_mode)
        self.autotable_render = AutoTableRenderer()

    def render_response(self, request, context, _route):
        if False:
            while True:
                i = 10
        try:
            from collections.abc import Sized
        except ImportError:
            from collections import Sized
        if isinstance(context, basestring):
            if self._guess_json(context):
                return Response(context, mimetype='application/json')
            elif '<html' in context[:168]:
                return Response(context, mimetype='text/html')
            else:
                return Response(context, mimetype='text/plain')
        if not isinstance(context, Sized):
            return Response(unicode(context), mimetype='text/plain')
        return self._serialize_to_resp(context, request, _route)
    __call__ = render_response

    def _serialize_to_resp(self, context, request, _route):
        if False:
            return 10
        req_format = request.args.get(self.qp_name)
        if req_format and req_format not in self._format_mime_map:
            raise ValueError('format expected one of %r, not %r' % (self.formats, req_format))
        resp_mime = self._format_mime_map.get(req_format)
        if not resp_mime and request.accept_mimetypes:
            resp_mime = request.accept_mimetypes.best_match(self.mimetypes)
        if resp_mime not in self._mime_format_map:
            resp_mime = self._default_mime
        if resp_mime == 'application/json':
            return self.json_render(context)
        elif resp_mime == 'text/html':
            return self.autotable_render(context, _route)
        return Response(unicode(context), mimetype='text/plain')

    @property
    def _mime_format_map(self):
        if False:
            print('Hello World!')
        return dict([(v, k) for (k, v) in self._format_mime_map.items()])

    @property
    def formats(self):
        if False:
            i = 10
            return i + 15
        return self._format_mime_map.keys()

    @property
    def mimetypes(self):
        if False:
            print('Hello World!')
        return self._format_mime_map.values()

    @staticmethod
    def _guess_json(text):
        if False:
            while True:
                i = 10
        if not text:
            return False
        elif text[0] == '{' and text[-1] == '}':
            return True
        elif text[0] == '[' and text[-1] == ']':
            return True
        else:
            return False

    @classmethod
    def factory(cls, *a, **kw):
        if False:
            for i in range(10):
                print('nop')

        def basic_render_factory(render_arg):
            if False:
                for i in range(10):
                    print('nop')
            return cls(*a, **kw)
        return basic_render_factory

def ident_ep(data):
    if False:
        print('Hello World!')
    return data

def main():
    if False:
        i = 10
        return i + 15
    rsc = {'data': _DATA}
    gpm = GetParamMiddleware('url')
    atr = AutoTableRenderer(max_depth=5)
    render_basic = BasicRender()
    app = Application([('/', ident_ep, render_basic), ('/json', ident_ep, render_basic), ('/fetch', fetch_json, render_basic)], rsc, [gpm])
    app.serve()
if __name__ == '__main__':
    main()