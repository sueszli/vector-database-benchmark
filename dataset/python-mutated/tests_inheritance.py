from unittest import TestCase
from django.test import tag

@tag('foo')
class FooBase(TestCase):
    pass

class Foo(FooBase):

    def test_no_new_tags(self):
        if False:
            while True:
                i = 10
        pass

    @tag('baz')
    def test_new_func_tag(self):
        if False:
            i = 10
            return i + 15
        pass

@tag('bar')
class FooBar(FooBase):

    def test_new_class_tag_only(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @tag('baz')
    def test_new_class_and_func_tags(self):
        if False:
            return 10
        pass