import win32con
from win32clipboard import *
if not __debug__:
    print('WARNING: The test code in this module uses assert')
    print('This instance of Python has asserts disabled, so many tests will be skipped')
cf_names = {}
for (name, val) in list(win32con.__dict__.items()):
    if name[:3] == 'CF_' and name != 'CF_SCREENFONTS':
        cf_names[val] = name

def TestEmptyClipboard():
    if False:
        while True:
            i = 10
    OpenClipboard()
    try:
        EmptyClipboard()
        assert EnumClipboardFormats(0) == 0, 'Clipboard formats were available after emptying it!'
    finally:
        CloseClipboard()

def TestText():
    if False:
        for i in range(10):
            print('nop')
    OpenClipboard()
    try:
        text = 'Hello from Python'
        text_bytes = text.encode('latin1')
        SetClipboardText(text)
        got = GetClipboardData(win32con.CF_TEXT)
        assert got == text_bytes, f"Didnt get the correct result back - '{got!r}'."
    finally:
        CloseClipboard()
    OpenClipboard()
    try:
        got = GetClipboardData(win32con.CF_UNICODETEXT)
        assert got == text, f"Didnt get the correct result back - '{got!r}'."
        assert isinstance(got, str), f"Didnt get the correct result back - '{got!r}'."
        got = GetClipboardData(win32con.CF_OEMTEXT)
        assert got == text_bytes, f"Didnt get the correct result back - '{got!r}'."
        EmptyClipboard()
        text = 'Hello from Python unicode'
        text_bytes = text.encode('latin1')
        SetClipboardData(win32con.CF_UNICODETEXT, text)
        got = GetClipboardData(win32con.CF_UNICODETEXT)
        assert got == text, f"Didnt get the correct result back - '{got!r}'."
        assert isinstance(got, str), f"Didnt get the correct result back - '{got!r}'."
    finally:
        CloseClipboard()
    OpenClipboard()
    try:
        got = GetClipboardData(win32con.CF_TEXT)
        assert got == text_bytes, f"Didnt get the correct result back - '{got!r}'."
        got = GetClipboardData(win32con.CF_UNICODETEXT)
        assert isinstance(got, str), f"Didnt get the correct result back - '{got!r}'."
        got = GetClipboardData(win32con.CF_OEMTEXT)
        assert got == text_bytes, f"Didnt get the correct result back - '{got!r}'."
        print('Clipboard text tests worked correctly')
    finally:
        CloseClipboard()

def TestClipboardEnum():
    if False:
        while True:
            i = 10
    OpenClipboard()
    try:
        enum = 0
        while 1:
            enum = EnumClipboardFormats(enum)
            if enum == 0:
                break
            assert IsClipboardFormatAvailable(enum), 'Have format, but clipboard says it is not available!'
            n = cf_names.get(enum, '')
            if not n:
                try:
                    n = GetClipboardFormatName(enum)
                except error:
                    n = f'unknown ({enum})'
            print('Have format', n)
        print('Clipboard enumerator tests worked correctly')
    finally:
        CloseClipboard()

class Foo:

    def __init__(self, **kw):
        if False:
            while True:
                i = 10
        self.__dict__.update(kw)

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__dict__ < other.__dict__

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.__dict__ == other.__dict__

def TestCustomFormat():
    if False:
        i = 10
        return i + 15
    OpenClipboard()
    try:
        fmt = RegisterClipboardFormat('Python Pickle Format')
        import pickle
        pickled_object = Foo(a=1, b=2, Hi=3)
        SetClipboardData(fmt, pickle.dumps(pickled_object))
        data = GetClipboardData(fmt)
        loaded_object = pickle.loads(data)
        assert pickle.loads(data) == pickled_object, 'Didnt get the correct data!'
        print('Clipboard custom format tests worked correctly')
    finally:
        CloseClipboard()
if __name__ == '__main__':
    TestEmptyClipboard()
    TestText()
    TestCustomFormat()
    TestClipboardEnum()
    TestEmptyClipboard()