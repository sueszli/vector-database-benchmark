import sys
import warnings
from functools import cached_property
from typing import Any
from pyodide.ffi import JsProxy
from pyscript import display, document, window
alert = window.alert

class BaseElement:

    def __init__(self, js_element):
        if False:
            while True:
                i = 10
        self._js = js_element
        self._parent = None
        self.style = StyleProxy(self)

    def __eq__(self, obj):
        if False:
            i = 10
            return i + 15
        'Check if the element is the same as the other element by comparing\n        the underlying JS element'
        return isinstance(obj, BaseElement) and obj._js == self._js

    @property
    def parent(self):
        if False:
            print('Hello World!')
        if self._parent:
            return self._parent
        if self._js.parentElement:
            self._parent = self.__class__(self._js.parentElement)
        return self._parent

    @property
    def __class(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__ if self.__class__ != PyDom else Element

    def create(self, type_, is_child=True, classes=None, html=None, label=None):
        if False:
            for i in range(10):
                print('nop')
        js_el = document.createElement(type_)
        element = self.__class(js_el)
        if classes:
            for class_ in classes:
                element.add_class(class_)
        if html is not None:
            element.html = html
        if label is not None:
            element.label = label
        if is_child:
            self.append(element)
        return element

    def find(self, selector):
        if False:
            return 10
        'Return an ElementCollection representing all the child elements that\n        match the specified selector.\n\n        Args:\n            selector (str): A string containing a selector expression\n\n        Returns:\n            ElementCollection: A collection of elements matching the selector\n        '
        elements = self._js.querySelectorAll(selector)
        if not elements:
            return None
        return ElementCollection([Element(el) for el in elements])

class Element(BaseElement):

    @property
    def children(self):
        if False:
            i = 10
            return i + 15
        return [self.__class__(el) for el in self._js.children]

    def append(self, child):
        if False:
            i = 10
            return i + 15
        if isinstance(child, JsProxy):
            return self.append(Element(child))
        elif isinstance(child, Element):
            self._js.appendChild(child._js)
            return child
        elif isinstance(child, ElementCollection):
            for el in child:
                self.append(el)

    @property
    def html(self):
        if False:
            i = 10
            return i + 15
        return self._js.innerHTML

    @html.setter
    def html(self, value):
        if False:
            print('Hello World!')
        self._js.innerHTML = value

    @property
    def content(self):
        if False:
            for i in range(10):
                print('nop')
        if self._js.tagName == 'TEMPLATE':
            warnings.warn('Content attribute not supported for template elements.', stacklevel=2)
            return None
        return self._js.innerHTML

    @content.setter
    def content(self, value):
        if False:
            for i in range(10):
                print('nop')
        if self._js.tagName == 'TEMPLATE':
            warnings.warn('Content attribute not supported for template elements.', stacklevel=2)
            return
        display(value, target=self.id)

    @property
    def id(self):
        if False:
            print('Hello World!')
        return self._js.id

    @id.setter
    def id(self, value):
        if False:
            return 10
        self._js.id = value

    @property
    def value(self):
        if False:
            return 10
        return self._js.value

    @value.setter
    def value(self, value):
        if False:
            i = 10
            return i + 15
        if not hasattr(self._js, 'value'):
            raise AttributeError(f'Element {self._js.tagName} has no value attribute. If you want to force a value attribute, set it directly using the `_js.value = <value>` javascript API attribute instead.')
        self._js.value = value

    def clone(self, new_id=None):
        if False:
            print('Hello World!')
        clone = Element(self._js.cloneNode(True))
        clone.id = new_id
        return clone

    def remove_class(self, classname):
        if False:
            return 10
        classList = self._js.classList
        if isinstance(classname, list):
            classList.remove(*classname)
        else:
            classList.remove(classname)
        return self

    def add_class(self, classname):
        if False:
            return 10
        classList = self._js.classList
        if isinstance(classname, list):
            classList.add(*classname)
        else:
            self._js.classList.add(classname)
        return self

    @property
    def classes(self):
        if False:
            i = 10
            return i + 15
        classes = self._js.classList.values()
        return [x for x in classes]

    def show_me(self):
        if False:
            for i in range(10):
                print('nop')
        self._js.scrollIntoView()

class StyleProxy(dict):

    def __init__(self, element: Element) -> None:
        if False:
            while True:
                i = 10
        self._element = element

    @cached_property
    def _style(self):
        if False:
            i = 10
            return i + 15
        return self._element._js.style

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self._style.getPropertyValue(key)

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self._style.setProperty(key, value)

    def remove(self, key):
        if False:
            for i in range(10):
                print('nop')
        self._style.removeProperty(key)

    def set(self, **kws):
        if False:
            print('Hello World!')
        for (k, v) in kws.items():
            self._element._js.style.setProperty(k, v)

    @property
    def visible(self):
        if False:
            while True:
                i = 10
        return self._element._js.style.visibility

    @visible.setter
    def visible(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._element._js.style.visibility = value

class StyleCollection:

    def __init__(self, collection: 'ElementCollection') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._collection = collection

    def __get__(self, obj, objtype=None):
        if False:
            print('Hello World!')
        return obj._get_attribute('style')

    def __getitem__(self, key):
        if False:
            return 10
        return self._collection._get_attribute('style')[key]

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        for element in self._collection._elements:
            element.style[key] = value

    def remove(self, key):
        if False:
            for i in range(10):
                print('nop')
        for element in self._collection._elements:
            element.style.remove(key)

class ElementCollection:

    def __init__(self, elements: [Element]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._elements = elements
        self.style = StyleCollection(self)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if isinstance(key, int):
            return self._elements[key]
        elif isinstance(key, slice):
            return ElementCollection(self._elements[key])
        elements = self._element.querySelectorAll(key)
        return ElementCollection([Element(el) for el in elements])

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._elements)

    def __eq__(self, obj):
        if False:
            i = 10
            return i + 15
        'Check if the element is the same as the other element by comparing\n        the underlying JS element'
        return isinstance(obj, ElementCollection) and obj._elements == self._elements

    def _get_attribute(self, attr, index=None):
        if False:
            i = 10
            return i + 15
        if index is None:
            return [getattr(el, attr) for el in self._elements]
        return getattr(self._elements[index], attr)

    def _set_attribute(self, attr, value):
        if False:
            while True:
                i = 10
        for el in self._elements:
            setattr(el, attr, value)

    @property
    def html(self):
        if False:
            while True:
                i = 10
        return self._get_attribute('html')

    @html.setter
    def html(self, value):
        if False:
            return 10
        self._set_attribute('html', value)

    @property
    def value(self):
        if False:
            while True:
                i = 10
        return self._get_attribute('value')

    @value.setter
    def value(self, value):
        if False:
            return 10
        self._set_attribute('value', value)

    @property
    def children(self):
        if False:
            print('Hello World!')
        return self._elements

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        yield from self._elements

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'{self.__class__.__name__} (length: {len(self._elements)}) {self._elements}'

class DomScope:

    def __getattr__(self, __name: str) -> Any:
        if False:
            i = 10
            return i + 15
        element = document[f'#{__name}']
        if element:
            return element[0]

class PyDom(BaseElement):
    BaseElement = BaseElement
    Element = Element
    ElementCollection = ElementCollection

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(document)
        self.ids = DomScope()
        self.body = Element(document.body)
        self.head = Element(document.head)

    def create(self, type_, classes=None, html=None):
        if False:
            i = 10
            return i + 15
        return super().create(type_, is_child=False, classes=classes, html=html)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if isinstance(key, int):
            indices = range(*key.indices(len(self.list)))
            return [self.list[i] for i in indices]
        elements = self._js.querySelectorAll(key)
        if not elements:
            return None
        return ElementCollection([Element(el) for el in elements])
dom = PyDom()
sys.modules[__name__] = dom