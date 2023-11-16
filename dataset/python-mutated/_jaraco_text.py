"""Functions brought over from jaraco.text.

These functions are not supposed to be used within `pip._internal`. These are
helper functions brought over from `jaraco.text` to enable vendoring newer
copies of `pkg_resources` without having to vendor `jaraco.text` and its entire
dependency cone; something that our vendoring setup is not currently capable of
handling.

License reproduced from original source below:

Copyright Jason R. Coombs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""
import functools
import itertools

def _nonblank(str):
    if False:
        for i in range(10):
            print('nop')
    return str and (not str.startswith('#'))

@functools.singledispatch
def yield_lines(iterable):
    if False:
        print('Hello World!')
    "\n    Yield valid lines of a string or iterable.\n\n    >>> list(yield_lines(''))\n    []\n    >>> list(yield_lines(['foo', 'bar']))\n    ['foo', 'bar']\n    >>> list(yield_lines('foo\\nbar'))\n    ['foo', 'bar']\n    >>> list(yield_lines('\\nfoo\\n#bar\\nbaz #comment'))\n    ['foo', 'baz #comment']\n    >>> list(yield_lines(['foo\\nbar', 'baz', 'bing\\n\\n\\n']))\n    ['foo', 'bar', 'baz', 'bing']\n    "
    return itertools.chain.from_iterable(map(yield_lines, iterable))

@yield_lines.register(str)
def _(text):
    if False:
        return 10
    return filter(_nonblank, map(str.strip, text.splitlines()))

def drop_comment(line):
    if False:
        for i in range(10):
            print('nop')
    "\n    Drop comments.\n\n    >>> drop_comment('foo # bar')\n    'foo'\n\n    A hash without a space may be in a URL.\n\n    >>> drop_comment('http://example.com/foo#bar')\n    'http://example.com/foo#bar'\n    "
    return line.partition(' #')[0]

def join_continuation(lines):
    if False:
        return 10
    "\n    Join lines continued by a trailing backslash.\n\n    >>> list(join_continuation(['foo \\\\', 'bar', 'baz']))\n    ['foobar', 'baz']\n    >>> list(join_continuation(['foo \\\\', 'bar', 'baz']))\n    ['foobar', 'baz']\n    >>> list(join_continuation(['foo \\\\', 'bar \\\\', 'baz']))\n    ['foobarbaz']\n\n    Not sure why, but...\n    The character preceeding the backslash is also elided.\n\n    >>> list(join_continuation(['goo\\\\', 'dly']))\n    ['godly']\n\n    A terrible idea, but...\n    If no line is available to continue, suppress the lines.\n\n    >>> list(join_continuation(['foo', 'bar\\\\', 'baz\\\\']))\n    ['foo']\n    "
    lines = iter(lines)
    for item in lines:
        while item.endswith('\\'):
            try:
                item = item[:-2].strip() + next(lines)
            except StopIteration:
                return
        yield item