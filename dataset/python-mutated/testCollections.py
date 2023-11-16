import sys
import unittest
import pythoncom
import win32com.client
import win32com.server.util
import win32com.test.util
import winerror
error = 'collection test error'

def MakeEmptyEnum():
    if False:
        return 10
    o = win32com.server.util.wrap(win32com.server.util.Collection())
    return win32com.client.Dispatch(o)

def MakeTestEnum():
    if False:
        i = 10
        return i + 15
    sub = win32com.server.util.wrap(win32com.server.util.Collection(['Sub1', 2, 'Sub3']))
    o = win32com.server.util.wrap(win32com.server.util.Collection([1, 'Two', 3, sub]))
    return win32com.client.Dispatch(o)

def TestEnumAgainst(o, check):
    if False:
        print('Hello World!')
    for i in range(len(check)):
        if o(i) != check[i]:
            raise error('Using default method gave the incorrect value - {}/{}'.format(repr(o(i)), repr(check[i])))
    for i in range(len(check)):
        if o.Item(i) != check[i]:
            raise error('Using Item method gave the incorrect value - {}/{}'.format(repr(o(i)), repr(check[i])))
    cmp = []
    for s in o:
        cmp.append(s)
    if cmp[:len(check)] != check:
        raise error('Result after looping isnt correct - {}/{}'.format(repr(cmp[:len(check)]), repr(check)))
    for i in range(len(check)):
        if o[i] != check[i]:
            raise error('Using indexing gave the incorrect value')

def TestEnum(quiet=None):
    if False:
        while True:
            i = 10
    if quiet is None:
        quiet = not '-v' in sys.argv
    if not quiet:
        print('Simple enum test')
    o = MakeTestEnum()
    check = [1, 'Two', 3]
    TestEnumAgainst(o, check)
    if not quiet:
        print('sub-collection test')
    sub = o[3]
    TestEnumAgainst(sub, ['Sub1', 2, 'Sub3'])
    o.Remove(o.Count() - 1)
    if not quiet:
        print('Remove item test')
    del check[1]
    o.Remove(1)
    TestEnumAgainst(o, check)
    if not quiet:
        print('Add item test')
    o.Add('New Item')
    check.append('New Item')
    TestEnumAgainst(o, check)
    if not quiet:
        print('Insert item test')
    o.Insert(2, -1)
    check.insert(2, -1)
    TestEnumAgainst(o, check)
    try:
        o()
        raise error('default method with no args worked when it shouldnt have!')
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_BADPARAMCOUNT:
            raise error(f'Expected DISP_E_BADPARAMCOUNT - got {exc}')
    try:
        o.Insert('foo', 2)
        raise error('Insert worked when it shouldnt have!')
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_TYPEMISMATCH:
            raise error(f'Expected DISP_E_TYPEMISMATCH - got {exc}')
    try:
        o.Remove(o.Count())
        raise error('Remove worked when it shouldnt have!')
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_BADINDEX:
            raise error(f'Expected DISP_E_BADINDEX - got {exc}')
    if not quiet:
        print('Empty collection test')
    o = MakeEmptyEnum()
    for item in o:
        raise error('Empty list performed an iteration')
    try:
        ob = o[1]
        raise error('Empty list could be indexed')
    except IndexError:
        pass
    try:
        ob = o[0]
        raise error('Empty list could be indexed')
    except IndexError:
        pass
    try:
        ob = o(0)
        raise error('Empty list could be indexed')
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_BADINDEX:
            raise error(f'Expected DISP_E_BADINDEX - got {exc}')

class TestCase(win32com.test.util.TestCase):

    def testEnum(self):
        if False:
            while True:
                i = 10
        TestEnum()
if __name__ == '__main__':
    unittest.main()