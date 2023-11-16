from django import forms
from telepath import Adapter, AdapterRegistry, JSContextBase
from wagtail.admin.staticfiles import versioned_static

class WagtailJSContextBase(JSContextBase):

    @property
    def base_media(self):
        if False:
            i = 10
            return i + 15
        return forms.Media(js=[versioned_static(self.telepath_js_path)])

class WagtailAdapterRegistry(AdapterRegistry):
    js_context_base_class = WagtailJSContextBase
registry = WagtailAdapterRegistry(telepath_js_path='wagtailadmin/js/telepath/telepath.js')
JSContext = registry.js_context_class

def register(adapter, cls):
    if False:
        print('Hello World!')
    registry.register(adapter, cls)

def adapter(js_constructor, base=Adapter):
    if False:
        print('Hello World!')
    "\n    Allows a class to implement its adapting logic with a `js_args()` method on the class itself.\n    This just helps reduce the amount of code you have to write.\n\n    For example:\n\n        @adapter('wagtail.mywidget')\n        class MyWidget():\n            ...\n\n            def js_args(self):\n                return [\n                    self.foo,\n                ]\n\n    Is equivalent to:\n\n        class MyWidget():\n            ...\n\n\n        class MyWidgetAdapter(Adapter):\n            js_constructor = 'wagtail.mywidget'\n\n            def js_args(self, obj):\n                return [\n                    self.foo,\n                ]\n    "

    def _wrapper(cls):
        if False:
            while True:
                i = 10
        ClassAdapter = type(cls.__name__ + 'Adapter', (base,), {'js_constructor': js_constructor, 'js_args': lambda self, obj: obj.js_args()})
        register(ClassAdapter(), cls)
        return cls
    return _wrapper