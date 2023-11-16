from r2.config.extensions import get_api_subtype
from r2.lib.utils import tup
from r2.lib.captcha import get_iden
from r2.lib.wrapped import Wrapped, StringTemplate
from r2.lib.filters import websafe_json, spaceCompress
from r2.lib.base import BaseController
from r2.lib.pages.things import wrap_links
from r2.models import IDBuilder, Listing
import simplejson
from pylons import tmpl_context as c
from pylons import app_globals as g

class JsonResponse(object):
    """
    Simple Api response handler, returning a list of errors generated
    in the api func's validators, as well as blobs of data set by the
    api func.
    """
    content_type = 'application/json'

    def __init__(self):
        if False:
            return 10
        self._clear()

    def _clear(self):
        if False:
            print('Hello World!')
        self._errors = set()
        self._new_captcha = False
        self._ratelimit = False
        self._data = {}

    def send_failure(self, error):
        if False:
            print('Hello World!')
        c.errors.add(error)
        self._clear()
        self._errors.add((error, None))

    def __call__(self, *a, **kw):
        if False:
            print('Hello World!')
        return self

    def __getattr__(self, key):
        if False:
            return 10
        return self

    def make_response(self):
        if False:
            print('Hello World!')
        res = {}
        if self._data:
            res['data'] = self._data
        if self._new_captcha:
            res['captcha'] = get_iden()
        if self._ratelimit:
            res['ratelimit'] = self._ratelimit
        res['errors'] = [(e[0], c.errors[e].message, e[1]) for e in self._errors]
        return {'json': res}

    def set_error(self, error_name, field_name):
        if False:
            for i in range(10):
                print('nop')
        self._errors.add((error_name, field_name))

    def has_error(self):
        if False:
            i = 10
            return i + 15
        return bool(self._errors)

    def has_errors(self, field_name, *errors, **kw):
        if False:
            return 10
        have_error = False
        field_name = tup(field_name)
        for error_name in errors:
            for fname in field_name:
                if (error_name, fname) in c.errors:
                    self.set_error(error_name, fname)
                    have_error = True
        return have_error

    def process_rendered(self, res):
        if False:
            for i in range(10):
                print('nop')
        return res

    def _things(self, things, action, *a, **kw):
        if False:
            print('Hello World!')
        '\n        function for inserting/replacing things in listings.\n        '
        things = tup(things)
        if not all((isinstance(t, Wrapped) for t in things)):
            wrap = kw.pop('wrap', Wrapped)
            things = wrap_links(things, wrapper=wrap)
        data = [self.process_rendered(t.render()) for t in things]
        if kw:
            for d in data:
                if d.has_key('data'):
                    d['data'].update(kw)
        self._data['things'] = data
        return data

    def insert_things(self, things, append=False, **kw):
        if False:
            for i in range(10):
                print('nop')
        return self._things(things, 'insert_things', append, **kw)

    def replace_things(self, things, keep_children=False, reveal=False, stubs=False, **kw):
        if False:
            return 10
        return self._things(things, 'replace_things', keep_children, reveal, stubs, **kw)

    def _send_data(self, **kw):
        if False:
            for i in range(10):
                print('nop')
        self._data.update(kw)

    def new_captcha(self):
        if False:
            return 10
        self._new_captcha = True

    def ratelimit(self, seconds):
        if False:
            for i in range(10):
                print('nop')
        self._ratelimit = seconds

class JQueryResponse(JsonResponse):
    """
    class which mimics the jQuery in javascript for allowing Dom
    manipulations on the client side.

    An instantiated JQueryResponse acts just like the "$" function on
    the JS layer with the exception of the ability to run arbitrary
    code on the client.  Selectors and method functions evaluate to
    new JQueryResponse objects, and the transformations are cataloged
    by the original object which can be iterated and sent across the
    wire.
    """

    def __init__(self, top_node=None):
        if False:
            for i in range(10):
                print('nop')
        if top_node:
            self.top_node = top_node
        else:
            self.top_node = self
        JsonResponse.__init__(self)
        self._clear()

    def _clear(self):
        if False:
            for i in range(10):
                print('nop')
        if self.top_node == self:
            self.objs = {self: 0}
            self.ops = []
        else:
            self.objs = None
            self.ops = None
        JsonResponse._clear(self)

    def process_rendered(self, res):
        if False:
            return 10
        if 'data' in res:
            if 'content' in res['data']:
                res['data']['content'] = spaceCompress(res['data']['content'])
        return res

    def send_failure(self, error):
        if False:
            for i in range(10):
                print('nop')
        c.errors.add(error)
        self._clear()
        self._errors.add((self, error, None))
        self.refresh()

    def __call__(self, *a):
        if False:
            i = 10
            return i + 15
        return self.top_node.transform(self, 'call', a)

    def __getattr__(self, key):
        if False:
            return 10
        if not key.startswith('__'):
            return self.top_node.transform(self, 'attr', key)

    def transform(self, obj, op, args):
        if False:
            return 10
        new = self.__class__(self)
        newi = self.objs[new] = len(self.objs)
        self.ops.append([self.objs[obj], newi, op, args])
        return new

    def set_error(self, error_name, field_name):
        if False:
            return 10
        self.top_node._errors.add((self, error_name, field_name))

    def has_error(self):
        if False:
            while True:
                i = 10
        return bool(self.top_node._errors)

    def make_response(self):
        if False:
            while True:
                i = 10
        for (form, error_name, field_name) in self._errors:
            selector = '.error.' + error_name
            if field_name:
                selector += '.field-' + field_name
            message = c.errors[error_name, field_name].message
            form.find(selector).show().text(message).end()
        return {'jquery': self.ops, 'success': not self.has_error()}

    def _things(self, things, action, *a, **kw):
        if False:
            i = 10
            return i + 15
        data = JsonResponse._things(self, things, action, *a, **kw)
        new = self.__getattr__(action)
        return new(data, *a)

    def insert_table_rows(self, rows, index=-1):
        if False:
            i = 10
            return i + 15
        new = self.__getattr__('insert_table_rows')
        return new([row.render(style='html') for row in tup(rows)], index)

    def new_captcha(self):
        if False:
            print('Hello World!')
        if not self._new_captcha:
            self.captcha(get_iden())
            self._new_captcha = True

    def get_input(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.find('*[name=%s]' % name)

    def set_inputs(self, **kw):
        if False:
            i = 10
            return i + 15
        for (k, v) in kw.iteritems():
            self.get_input(k).val(v).end()
        return self

    def focus_input(self, name):
        if False:
            return 10
        return self.get_input(name).focus().end()

    def set_html(self, selector, value):
        if False:
            while True:
                i = 10
        if value:
            return self.find(selector).show().html(value).end()
        return self.find(selector).hide().html('').end()

    def set_text(self, selector, value):
        if False:
            for i in range(10):
                print('nop')
        if value:
            return self.find(selector).show().text(value).end()
        return self.find(selector).hide().html('').end()

    def set(self, **kw):
        if False:
            return 10
        obj = self
        for (k, v) in kw.iteritems():
            obj = obj.attr(k, v)
        return obj

    def refresh(self):
        if False:
            while True:
                i = 10
        return self.top_node.transform(self, 'refresh', [])