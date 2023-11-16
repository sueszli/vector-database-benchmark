"""
Tests for L{twisted.web._stan} portion of the L{twisted.web.template}
implementation.
"""
import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag

def proto(*a: Flattenable, **kw: Flattenable) -> Tag:
    if False:
        print('Hello World!')
    '\n    Produce a new tag for testing.\n    '
    return Tag('hello')(*a, **kw)

class TagTests(TestCase):
    """
    Tests for L{Tag}.
    """

    def test_renderAttribute(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Setting an attribute named C{render} will change the C{render} instance\n        variable instead of adding an attribute.\n        '
        tag = proto(render='myRenderer')
        self.assertEqual(tag.render, 'myRenderer')
        self.assertEqual(tag.attributes, {})

    def test_renderAttributeNonString(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Attempting to set an attribute named C{render} to something other than\n        a string will raise L{TypeError}.\n        '
        with self.assertRaises(TypeError) as e:
            proto(render=83)
        self.assertEqual(e.exception.args[0], 'Value for "render" attribute must be str, got 83')

    def test_fillSlots(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Tag.fillSlots} returns self.\n        '
        tag = proto()
        self.assertIdentical(tag, tag.fillSlots(test='test'))

    def test_cloneShallow(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{Tag.clone} copies all attributes and children of a tag, including its\n        render attribute.  If the shallow flag is C{False}, that's where it\n        stops.\n        "
        innerList = ['inner list']
        tag = proto('How are you', innerList, hello='world', render='aSampleMethod')
        tag.fillSlots(foo='bar')
        tag.filename = 'foo/bar'
        tag.lineNumber = 6
        tag.columnNumber = 12
        clone = tag.clone(deep=False)
        self.assertEqual(clone.attributes['hello'], 'world')
        self.assertNotIdentical(clone.attributes, tag.attributes)
        self.assertEqual(clone.children, ['How are you', innerList])
        self.assertNotIdentical(clone.children, tag.children)
        self.assertIdentical(clone.children[1], innerList)
        self.assertEqual(tag.slotData, clone.slotData)
        self.assertNotIdentical(tag.slotData, clone.slotData)
        self.assertEqual(clone.filename, 'foo/bar')
        self.assertEqual(clone.lineNumber, 6)
        self.assertEqual(clone.columnNumber, 12)
        self.assertEqual(clone.render, 'aSampleMethod')

    def test_cloneDeep(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{Tag.clone} copies all attributes and children of a tag, including its\n        render attribute.  In its normal operating mode (where the deep flag is\n        C{True}, as is the default), it will clone all sub-lists and sub-tags.\n        '
        innerTag = proto('inner')
        innerList = ['inner list']
        tag = proto('How are you', innerTag, innerList, hello='world', render='aSampleMethod')
        tag.fillSlots(foo='bar')
        tag.filename = 'foo/bar'
        tag.lineNumber = 6
        tag.columnNumber = 12
        clone = tag.clone()
        self.assertEqual(clone.attributes['hello'], 'world')
        self.assertNotIdentical(clone.attributes, tag.attributes)
        self.assertNotIdentical(clone.children, tag.children)
        self.assertIdentical(tag.children[1], innerTag)
        self.assertNotIdentical(clone.children[1], innerTag)
        self.assertIdentical(tag.children[2], innerList)
        self.assertNotIdentical(clone.children[2], innerList)
        self.assertEqual(tag.slotData, clone.slotData)
        self.assertNotIdentical(tag.slotData, clone.slotData)
        self.assertEqual(clone.filename, 'foo/bar')
        self.assertEqual(clone.lineNumber, 6)
        self.assertEqual(clone.columnNumber, 12)
        self.assertEqual(clone.render, 'aSampleMethod')

    def test_cloneGeneratorDeprecation(self) -> None:
        if False:
            print('Hello World!')
        '\n        Cloning a tag containing a generator is unsafe. To avoid breaking\n        programs that only flatten the clone or only flatten the original,\n        we deprecate old behavior rather than making it an error immediately.\n        '
        tag = proto((str(n) for n in range(10)))
        self.assertWarns(DeprecationWarning, 'Cloning a Tag which contains a generator is unsafe, since the generator can be consumed only once; this is deprecated since Twisted 21.7.0 and will raise an exception in the future', sys.modules[Tag.__module__].__file__, tag.clone)

    def test_cloneCoroutineDeprecation(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Cloning a tag containing a coroutine is unsafe. To avoid breaking\n        programs that only flatten the clone or only flatten the original,\n        we deprecate old behavior rather than making it an error immediately.\n        '

        async def asyncFunc() -> NoReturn:
            raise NotImplementedError
        coro = asyncFunc()
        tag = proto('123', coro, '789')
        try:
            self.assertWarns(DeprecationWarning, 'Cloning a Tag which contains a coroutine is unsafe, since the coroutine can run only once; this is deprecated since Twisted 21.7.0 and will raise an exception in the future', sys.modules[Tag.__module__].__file__, tag.clone)
        finally:
            coro.close()

    def test_clear(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{Tag.clear} removes all children from a tag, but leaves its attributes\n        in place.\n        '
        tag = proto('these are', 'children', 'cool', andSoIs='this-attribute')
        tag.clear()
        self.assertEqual(tag.children, [])
        self.assertEqual(tag.attributes, {'andSoIs': 'this-attribute'})

    def test_suffix(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{Tag.__call__} accepts Python keywords with a suffixed underscore as\n        the DOM attribute of that literal suffix.\n        '
        proto = Tag('div')
        tag = proto()
        tag(class_='a')
        self.assertEqual(tag.attributes, {'class': 'a'})

    def test_commentReprPy3(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{Comment.__repr__} returns a value which makes it easy to see what's\n        in the comment.\n        "
        self.assertEqual(repr(Comment('hello there')), "Comment('hello there')")

    def test_cdataReprPy3(self) -> None:
        if False:
            while True:
                i = 10
        "\n        L{CDATA.__repr__} returns a value which makes it easy to see what's in\n        the comment.\n        "
        self.assertEqual(repr(CDATA('test data')), "CDATA('test data')")

    def test_charrefRepr(self) -> None:
        if False:
            return 10
        '\n        L{CharRef.__repr__} returns a value which makes it easy to see what\n        character is referred to.\n        '
        snowman = ord('☃')
        self.assertEqual(repr(CharRef(snowman)), 'CharRef(9731)')