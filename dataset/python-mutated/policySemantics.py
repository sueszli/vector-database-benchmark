import unittest
import pythoncom
import win32com.client
import win32com.server.util
import win32com.test.util
import winerror

class Error(Exception):
    pass

class PythonSemanticClass:
    _public_methods_ = ['In']
    _dispid_to_func_ = {10: 'Add', 11: 'Remove'}

    def __init__(self):
        if False:
            return 10
        self.list = []

    def _NewEnum(self):
        if False:
            return 10
        return win32com.server.util.NewEnum(self.list)

    def _value_(self):
        if False:
            i = 10
            return i + 15
        return self.list

    def _Evaluate(self):
        if False:
            i = 10
            return i + 15
        return sum(self.list)

    def In(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value in self.list

    def Add(self, value):
        if False:
            return 10
        self.list.append(value)

    def Remove(self, value):
        if False:
            while True:
                i = 10
        self.list.remove(value)

def DispExTest(ob):
    if False:
        i = 10
        return i + 15
    if not __debug__:
        print('WARNING: Tests dressed up as assertions are being skipped!')
    assert ob.GetDispID('Add', 0) == 10, 'Policy did not honour the dispid'
    assert ob.GetDispID('Remove', 0) == 11, 'Policy did not honour the dispid'
    assert ob.GetDispID('In', 0) == 1000, 'Allocated dispid unexpected value'
    assert ob.GetDispID('_NewEnum', 0) == pythoncom.DISPID_NEWENUM, '_NewEnum() got unexpected DISPID'
    dispids = []
    dispid = -1
    while 1:
        try:
            dispid = ob.GetNextDispID(0, dispid)
            dispids.append(dispid)
        except pythoncom.com_error as xxx_todo_changeme:
            (hr, desc, exc, arg) = xxx_todo_changeme.args
            assert hr == winerror.S_FALSE, 'Bad result at end of enum'
            break
    dispids.sort()
    if dispids != [pythoncom.DISPID_EVALUATE, pythoncom.DISPID_NEWENUM, 10, 11, 1000]:
        raise Error('Got back the wrong dispids: %s' % dispids)

def SemanticTest(ob):
    if False:
        for i in range(10):
            print('nop')
    ob.Add(1)
    ob.Add(2)
    ob.Add(3)
    if ob() != (1, 2, 3):
        raise Error('Bad result - got %s' % repr(ob()))
    dispob = ob._oleobj_
    rc = dispob.Invoke(pythoncom.DISPID_EVALUATE, 0, pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET, 1)
    if rc != 6:
        raise Error('Evaluate returned %d' % rc)

class Tester(win32com.test.util.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        debug = 0
        import win32com.server.dispatcher
        if debug:
            dispatcher = win32com.server.dispatcher.DefaultDebugDispatcher
        else:
            dispatcher = None
        disp = win32com.server.util.wrap(PythonSemanticClass(), useDispatcher=dispatcher)
        self.ob = win32com.client.Dispatch(disp)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.ob = None

    def testSemantics(self):
        if False:
            for i in range(10):
                print('nop')
        SemanticTest(self.ob)

    def testIDispatchEx(self):
        if False:
            for i in range(10):
                print('nop')
        dispexob = self.ob._oleobj_.QueryInterface(pythoncom.IID_IDispatchEx)
        DispExTest(dispexob)
if __name__ == '__main__':
    unittest.main()