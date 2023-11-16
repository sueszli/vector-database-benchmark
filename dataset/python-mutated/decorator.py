"""
*What is this pattern about?
The Decorator pattern is used to dynamically add a new feature to an
object without changing its implementation. It differs from
inheritance because the new feature is added only to that particular
object, not to the entire subclass.

*What does this example do?
This example shows a way to add formatting options (boldface and
italic) to a text by appending the corresponding tags (<b> and
<i>). Also, we can see that decorators can be applied one after the other,
since the original text is passed to the bold wrapper, which in turn
is passed to the italic wrapper.

*Where is the pattern used practically?
The Grok framework uses decorators to add functionalities to methods,
like permissions or subscription to an event:
http://grok.zope.org/doc/current/reference/decorators.html

*References:
https://sourcemaking.com/design_patterns/decorator

*TL;DR
Adds behaviour to object without affecting its class.
"""

class TextTag:
    """Represents a base text tag"""

    def __init__(self, text: str) -> None:
        if False:
            print('Hello World!')
        self._text = text

    def render(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._text

class BoldWrapper(TextTag):
    """Wraps a tag in <b>"""

    def __init__(self, wrapped: TextTag) -> None:
        if False:
            print('Hello World!')
        self._wrapped = wrapped

    def render(self) -> str:
        if False:
            print('Hello World!')
        return f'<b>{self._wrapped.render()}</b>'

class ItalicWrapper(TextTag):
    """Wraps a tag in <i>"""

    def __init__(self, wrapped: TextTag) -> None:
        if False:
            print('Hello World!')
        self._wrapped = wrapped

    def render(self) -> str:
        if False:
            while True:
                i = 10
        return f'<i>{self._wrapped.render()}</i>'

def main():
    if False:
        i = 10
        return i + 15
    '\n    >>> simple_hello = TextTag("hello, world!")\n    >>> special_hello = ItalicWrapper(BoldWrapper(simple_hello))\n\n    >>> print("before:", simple_hello.render())\n    before: hello, world!\n\n    >>> print("after:", special_hello.render())\n    after: <i><b>hello, world!</b></i>\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()