"""Documentation control decorators."""
from typing import Optional, TypeVar
T = TypeVar('T')
_DEPRECATED = '_tf_docs_deprecated'

def set_deprecated(obj: T) -> T:
    if False:
        return 10
    'Explicitly tag an object as deprecated for the doc generator.'
    setattr(obj, _DEPRECATED, None)
    return obj
_INHERITABLE_HEADER = '_tf_docs_inheritable_header'

def inheritable_header(text):
    if False:
        return 10

    def _wrapped(cls):
        if False:
            return 10
        setattr(cls, _INHERITABLE_HEADER, text)
        return cls
    return _wrapped

def get_inheritable_header(obj) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    return getattr(obj, _INHERITABLE_HEADER, None)
header = inheritable_header
get_header = get_inheritable_header
_DO_NOT_DOC = '_tf_docs_do_not_document'

def do_not_generate_docs(obj: T) -> T:
    if False:
        while True:
            i = 10
    'A decorator: Do not generate docs for this object.\n\n  For example the following classes:\n\n  ```\n  class Parent(object):\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child(Parent):\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n  ```\n\n  Produce the following api_docs:\n\n  ```\n  /Parent.md\n    # method1\n    # method2\n  /Child.md\n    # method1\n    # method2\n  ```\n\n  This decorator allows you to skip classes or methods:\n\n  ```\n  @do_not_generate_docs\n  class Parent(object):\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child(Parent):\n    @do_not_generate_docs\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n  ```\n\n  This will only produce the following docs:\n\n  ```\n  /Child.md\n    # method2\n  ```\n\n  Note: This is implemented by adding a hidden attribute on the object, so it\n  cannot be used on objects which do not allow new attributes to be added. So\n  this decorator must go *below* `@property`, `@classmethod`,\n  or `@staticmethod`:\n\n  ```\n  class Example(object):\n    @property\n    @do_not_generate_docs\n    def x(self):\n      return self._x\n  ```\n\n  Args:\n    obj: The object to hide from the generated docs.\n\n  Returns:\n    obj\n  '
    setattr(obj, _DO_NOT_DOC, None)
    return obj
_DO_NOT_DOC_INHERITABLE = '_tf_docs_do_not_doc_inheritable'

def do_not_doc_inheritable(obj: T) -> T:
    if False:
        while True:
            i = 10
    'A decorator: Do not generate docs for this method.\n\n  This version of the decorator is "inherited" by subclasses. No docs will be\n  generated for the decorated method in any subclass. Even if the sub-class\n  overrides the method.\n\n  For example, to ensure that `method1` is **never documented** use this\n  decorator on the base-class:\n\n  ```\n  class Parent(object):\n    @do_not_doc_inheritable\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child(Parent):\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n  ```\n  This will produce the following docs:\n\n  ```\n  /Parent.md\n    # method2\n  /Child.md\n    # method2\n  ```\n\n  When generating docs for a class\'s arributes, the `__mro__` is searched and\n  the attribute will be skipped if this decorator is detected on the attribute\n  on any class in the `__mro__`.\n\n  Note: This is implemented by adding a hidden attribute on the object, so it\n  cannot be used on objects which do not allow new attributes to be added. So\n  this decorator must go *below* `@property`, `@classmethod`,\n  or `@staticmethod`:\n\n  ```\n  class Example(object):\n    @property\n    @do_not_doc_inheritable\n    def x(self):\n      return self._x\n  ```\n\n  Args:\n    obj: The class-attribute to hide from the generated docs.\n\n  Returns:\n    obj\n  '
    setattr(obj, _DO_NOT_DOC_INHERITABLE, None)
    return obj
_FOR_SUBCLASS_IMPLEMENTERS = '_tf_docs_tools_for_subclass_implementers'

def for_subclass_implementers(obj: T) -> T:
    if False:
        while True:
            i = 10
    "A decorator: Only generate docs for this method in the defining class.\n\n  Also group this method's docs with and `@abstractmethod` in the class's docs.\n\n  No docs will generated for this class attribute in sub-classes.\n\n  The canonical use case for this is `tf.keras.layers.Layer.call`: It's a\n  public method, essential for anyone implementing a subclass, but it should\n  never be called directly.\n\n  Works on method, or other class-attributes.\n\n  When generating docs for a class's arributes, the `__mro__` is searched and\n  the attribute will be skipped if this decorator is detected on the attribute\n  on any **parent** class in the `__mro__`.\n\n  For example:\n\n  ```\n  class Parent(object):\n    @for_subclass_implementers\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child1(Parent):\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child2(Parent):\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n  ```\n\n  This will produce the following docs:\n\n  ```\n  /Parent.md\n    # method1\n    # method2\n  /Child1.md\n    # method2\n  /Child2.md\n    # method2\n  ```\n\n  Note: This is implemented by adding a hidden attribute on the object, so it\n  cannot be used on objects which do not allow new attributes to be added. So\n  this decorator must go *below* `@property`, `@classmethod`,\n  or `@staticmethod`:\n\n  ```\n  class Example(object):\n    @property\n    @for_subclass_implementers\n    def x(self):\n      return self._x\n  ```\n\n  Args:\n    obj: The class-attribute to hide from the generated docs.\n\n  Returns:\n    obj\n  "
    setattr(obj, _FOR_SUBCLASS_IMPLEMENTERS, None)
    return obj
do_not_doc_in_subclasses = for_subclass_implementers
_DOC_PRIVATE = '_tf_docs_doc_private'

def doc_private(obj: T) -> T:
    if False:
        return 10
    'A decorator: Generates docs for private methods/functions.\n\n  For example:\n\n  ```\n  class Try:\n\n    @doc_controls.doc_private\n    def _private(self):\n      ...\n  ```\n\n  As a rule of thumb, private(beginning with `_`) methods/functions are\n  not documented.\n\n  This decorator allows to force document a private method/function.\n\n  Args:\n    obj: The class-attribute to hide from the generated docs.\n\n  Returns:\n    obj\n  '
    setattr(obj, _DOC_PRIVATE, None)
    return obj
_DOC_IN_CURRENT_AND_SUBCLASSES = '_tf_docs_doc_in_current_and_subclasses'

def doc_in_current_and_subclasses(obj: T) -> T:
    if False:
        return 10
    "Overrides `do_not_doc_in_subclasses` decorator.\n\n  If this decorator is set on a child class's method whose parent's method\n  contains `do_not_doc_in_subclasses`, then that will be overriden and the\n  child method will get documented. All classes inherting from the child will\n  also document that method.\n\n  For example:\n\n  ```\n  class Parent:\n    @do_not_doc_in_subclasses\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child1(Parent):\n    @doc_in_current_and_subclasses\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child2(Parent):\n    def method1(self):\n      pass\n    def method2(self):\n      pass\n\n  class Child11(Child1):\n    pass\n  ```\n\n  This will produce the following docs:\n\n  ```\n  /Parent.md\n    # method1\n    # method2\n  /Child1.md\n    # method1\n    # method2\n  /Child2.md\n    # method2\n  /Child11.md\n    # method1\n    # method2\n  ```\n\n  Args:\n    obj: The class-attribute to hide from the generated docs.\n\n  Returns:\n    obj\n  "
    setattr(obj, _DOC_IN_CURRENT_AND_SUBCLASSES, None)
    return obj