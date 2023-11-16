"""Library for ``libdoc.html`` testing.

*URL:*    http://robotframework.org

_Image:_  https://github.com/robotframework/visual-identity/raw/master/logo/robot-framework.svg

_*Cross linking*_: `Links`, `One Paragraph`, `HR`, `hr`.
`section`, `Nön-ÄSCÏÏ`, `Special ½!"#¤%&/()=?<|>+-_.!~*'() chars`

----------------------------

= Section =
== Subsection with Ääkköset ==

| *My* | *Table* |
| 1    | 2       |
| foo  |
regular line
| block formatted
|    content		and whitespaces
"""
from datetime import date
from enum import Enum
from typing import TypedDict, Union
from robot.api.deco import keyword, not_keyword
not_keyword(TypedDict)

@not_keyword
def parse_date(value: str):
    if False:
        while True:
            i = 10
    'Date in format ``dd.mm.yyyy``.'
    (d, m, y) = [int(v) for v in value.split('.')]
    return date(y, m, d)

class Direction(Enum):
    """Move direction."""
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Point(TypedDict):
    """Pointless point."""
    x: int
    y: int

class date2(date):
    pass
ROBOT_LIBRARY_CONVERTERS = {date: parse_date}

def type_hints(a: int, b: Direction, c: Point, d: date, e: bool=True, f: Union[int, date]=None):
    if False:
        i = 10
        return i + 15
    'We use `integer`, `date`, `Direction`, and many other types.'
    pass

def type_aliases(a: date, b: date2):
    if False:
        print('Hello World!')
    pass

def int10(i: int):
    if False:
        while True:
            i = 10
    pass
int1 = int2 = int3 = int4 = int5 = int6 = int7 = int8 = int9 = int10

def one_paragraph(one):
    if False:
        while True:
            i = 10
    'Hello, world!'

def multiple_paragraphs(one, two, three='default'):
    if False:
        return 10
    'Hello, world!\n\n    Second paragraph *has formatting* and [http://example.com|link].\n    It also refers to argument ``one`` using ``code`` style.\n    This is still part of second paragraph.\n\n    Third paragraph is _short_.\n\n    Tags: tag, another tag\n    '

def tables_alone():
    if False:
        print('Hello World!')
    '\n    | *a* | *b*   | *c*  |\n    | 1st | table | here |\n\n    | 2nd | table | has | only | one | row |\n\n    Tags: another tag\n    '

def preformatted():
    if False:
        for i in range(10):
            print('nop')
    '\n    | First block\n    | has two lines\n\n    | Second has only one\n\n    Tags: TAG\n    '

def lists(*list):
    if False:
        i = 10
        return i + 15
    '\n    - first\n    - second\n\n    - another\n    '

def hr():
    if False:
        return 10
    '\n    ---\n    ---\n\n    ---------------\n    '

def links():
    if False:
        for i in range(10):
            print('nop')
    '\n    - `Lists`, `One Paragraph`, `HR`, `hr`, `nön-äscïï`, `Special ½!"#¤%&/()=?<|>+-_.!~*\'() chars`\n    - `Section`, `Sub section with ääkköset`\n    - `Shortcuts`, `keywords`, `LIBRARY intro duct ion`\n    - http://robotframework.org\n    - [http://robotframework.org|Robot Framework]\n    '

def images():
    if False:
        i = 10
        return i + 15
    '\n    https://github.com/robotframework/visual-identity/raw/master/logo/robot-framework.svg\n\n    Images [https://github.com/robotframework/visual-identity/raw/master/logo/robot-framework.svg|title]\n    inside paragraphs. This one is also a link:\n    [https://github.com/robotframework/visual-identity/raw/master/logo/robot-framework.svg|\n    https://github.com/robotframework/visual-identity/raw/master/logo/robot-framework.svg]\n    '

@keyword('Nön-ÄSCÏÏ', tags=['Nön', 'äscïï', 'tägß'])
def non_ascii(ärg='ööööö'):
    if False:
        for i in range(10):
            print('nop')
    'Älsö döc häs nön-äscïï stüff. Ïnclüdïng ☃.'

@keyword('Special ½!"#¤%&/()=?<|>+-_.!~*\'() chars', tags=['½!"#¤%&/()=?', "<|>+-_.!~*'()"])
def special_chars():
    if False:
        i = 10
        return i + 15
    ' Also doc has ½!"#¤%&/()=?<|>+-_.!~*\'().'

def zzz_long_documentation():
    if False:
        print('Hello World!')
    '\n    Last keyword has a bit longer documentation to make sure page moves\n    when testing linking to keywords.\n\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    - - -\n    '