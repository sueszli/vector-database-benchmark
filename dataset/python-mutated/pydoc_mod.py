"""This is a test module for test_pydoc"""
import types
import typing
__author__ = 'Benjamin Peterson'
__credits__ = 'Nobody'
__version__ = '1.2.3.4'
__xyz__ = 'X, Y and Z'

class A:
    """Hello and goodbye"""

    def __init__():
        if False:
            for i in range(10):
                print('nop')
        'Wow, I have no function!'
        pass

class B(object):
    NO_MEANING: str = 'eggs'
    pass

class C(object):

    def say_no(self):
        if False:
            i = 10
            return i + 15
        return 'no'

    def get_answer(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return say_no() '
        return self.say_no()

    def is_it_true(self):
        if False:
            return 10
        ' Return self.get_answer() '
        return self.get_answer()

    def __class_getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return types.GenericAlias(self, item)

def doc_func():
    if False:
        while True:
            i = 10
    "\n    This function solves all of the world's problems:\n    hunger\n    lack of Python\n    war\n    "

def nodoc_func():
    if False:
        i = 10
        return i + 15
    pass
list_alias1 = typing.List[int]
list_alias2 = list[int]
c_alias = C[int]
type_union1 = typing.Union[int, str]
type_union2 = int | str