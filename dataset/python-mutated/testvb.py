import traceback
import pythoncom
import win32com.client
import win32com.client.dynamic
import win32com.client.gencache
import winerror
from win32com.server.util import wrap
from win32com.test import util
useDispatcher = None
error = RuntimeError

class TestObject:
    _public_methods_ = ['CallbackVoidOneByRef', 'CallbackResultOneByRef', 'CallbackVoidTwoByRef', 'CallbackString', 'CallbackResultOneByRefButReturnNone', 'CallbackVoidOneByRefButReturnNone', 'CallbackArrayResult', 'CallbackArrayResultOneArrayByRef', 'CallbackArrayResultWrongSize']

    def CallbackVoidOneByRef(self, intVal):
        if False:
            i = 10
            return i + 15
        return intVal + 1

    def CallbackResultOneByRef(self, intVal):
        if False:
            for i in range(10):
                print('nop')
        return (intVal, intVal + 1)

    def CallbackVoidTwoByRef(self, int1, int2):
        if False:
            print('Hello World!')
        return (int1 + int2, int1 - int2)

    def CallbackString(self, strVal):
        if False:
            print('Hello World!')
        return (0, strVal + ' has visited Python')

    def CallbackArrayResult(self, arrayVal):
        if False:
            while True:
                i = 10
        ret = []
        for i in arrayVal:
            ret.append(i + 1)
        return ret

    def CallbackArrayResultWrongSize(self, arrayVal):
        if False:
            i = 10
            return i + 15
        return list(arrayVal[:-1])

    def CallbackArrayResultOneArrayByRef(self, arrayVal):
        if False:
            while True:
                i = 10
        ret = []
        for i in arrayVal:
            ret.append(i + 1)
        return (list(arrayVal), ret)

    def CallbackResultOneByRefButReturnNone(self, intVal):
        if False:
            for i in range(10):
                print('nop')
        return

    def CallbackVoidOneByRefButReturnNone(self, intVal):
        if False:
            while True:
                i = 10
        return

def TestVB(vbtest, bUseGenerated):
    if False:
        return 10
    vbtest.LongProperty = -1
    if vbtest.LongProperty != -1:
        raise error('Could not set the long property correctly.')
    vbtest.IntProperty = 10
    if vbtest.IntProperty != 10:
        raise error('Could not set the integer property correctly.')
    vbtest.VariantProperty = 10
    if vbtest.VariantProperty != 10:
        raise error('Could not set the variant integer property correctly.')
    vbtest.VariantProperty = memoryview(b'raw\x00data')
    if vbtest.VariantProperty != memoryview(b'raw\x00data'):
        raise error('Could not set the variant buffer property correctly.')
    vbtest.StringProperty = 'Hello from Python'
    if vbtest.StringProperty != 'Hello from Python':
        raise error('Could not set the string property correctly.')
    vbtest.VariantProperty = 'Hello from Python'
    if vbtest.VariantProperty != 'Hello from Python':
        raise error('Could not set the variant string property correctly.')
    vbtest.VariantProperty = (1.0, 2.0, 3.0)
    if vbtest.VariantProperty != (1.0, 2.0, 3.0):
        raise error("Could not set the variant property to an array of floats correctly - '{}'.".format(vbtest.VariantProperty))
    TestArrays(vbtest, bUseGenerated)
    TestStructs(vbtest)
    TestCollections(vbtest)
    assert vbtest.TakeByValObject(vbtest) == vbtest
    if bUseGenerated:
        ob = vbtest.TakeByRefObject(vbtest)
        assert ob[0] == vbtest and ob[1] == vbtest
        vbtest.VariantPutref = vbtest
        if vbtest.VariantPutref._oleobj_ != vbtest._oleobj_:
            raise error('Could not set the VariantPutref property correctly.')
        if vbtest.IncrementIntegerParam(1) != 2:
            raise error('Could not pass an integer byref')
        if vbtest.IncrementVariantParam(1) != 2:
            raise error('Could not pass an int VARIANT byref:' + str(vbtest.IncrementVariantParam(1)))
        if vbtest.IncrementVariantParam(1.5) != 2.5:
            raise error('Could not pass a float VARIANT byref')
        callback_ob = wrap(TestObject(), useDispatcher=useDispatcher)
        vbtest.DoSomeCallbacks(callback_ob)
    ret = vbtest.PassIntByVal(1)
    if ret != 2:
        raise error('Could not increment the integer - ' + str(ret))
    TestVBInterface(vbtest)
    if bUseGenerated:
        ret = vbtest.PassIntByRef(1)
        if ret != (1, 2):
            raise error('Could not increment the integer - ' + str(ret))

def _DoTestCollection(vbtest, col_name, expected):
    if False:
        i = 10
        return i + 15

    def _getcount(ob):
        if False:
            for i in range(10):
                print('nop')
        r = getattr(ob, 'Count')
        if isinstance(r, Callable):
            return r()
        return r
    c = getattr(vbtest, col_name)
    check = []
    for item in c:
        check.append(item)
    if check != list(expected):
        raise error(f"Collection {col_name} didn't have {expected!r} (had {check!r})")
    check = []
    for item in c:
        check.append(item)
    if check != list(expected):
        raise error("Collection 2nd time around {} didn't have {!r} (had {!r})".format(col_name, expected, check))
    i = iter(getattr(vbtest, col_name))
    check = []
    for item in i:
        check.append(item)
    if check != list(expected):
        raise error("Collection iterator {} didn't have {!r} 2nd time around (had {!r})".format(col_name, expected, check))
    check = []
    for item in i:
        check.append(item)
    if check != []:
        raise error("2nd time around Collection iterator {} wasn't empty (had {!r})".format(col_name, check))
    c = getattr(vbtest, col_name)
    if len(c) != _getcount(c):
        raise error(f"Collection {col_name} __len__({len(c)!r}) wasn't==Count({_getcount(c)!r})")
    c = getattr(vbtest, col_name)
    check = []
    for i in range(_getcount(c)):
        check.append(c[i])
    if check != list(expected):
        raise error(f"Collection {col_name} didn't have {expected!r} (had {check!r})")
    c = getattr(vbtest, col_name)._NewEnum()
    check = []
    while 1:
        n = c.Next()
        if not n:
            break
        check.append(n[0])
    if check != list(expected):
        raise error(f"Collection {col_name} didn't have {expected!r} (had {check!r})")

def TestCollections(vbtest):
    if False:
        i = 10
        return i + 15
    _DoTestCollection(vbtest, 'CollectionProperty', [1, 'Two', '3'])
    if vbtest.CollectionProperty[0] != 1:
        raise error('The CollectionProperty[0] element was not the default value')
    _DoTestCollection(vbtest, 'EnumerableCollectionProperty', [])
    vbtest.EnumerableCollectionProperty.Add(1)
    vbtest.EnumerableCollectionProperty.Add('Two')
    vbtest.EnumerableCollectionProperty.Add('3')
    _DoTestCollection(vbtest, 'EnumerableCollectionProperty', [1, 'Two', '3'])

def _DoTestArray(vbtest, data, expected_exception=None):
    if False:
        i = 10
        return i + 15
    try:
        vbtest.ArrayProperty = data
        if expected_exception is not None:
            raise error("Expected '%s'" % expected_exception)
    except expected_exception:
        return
    got = vbtest.ArrayProperty
    if got != data:
        raise error(f'Could not set the array data correctly - got {got!r}, expected {data!r}')

def TestArrays(vbtest, bUseGenerated):
    if False:
        while True:
            i = 10
    _DoTestArray(vbtest, ())
    _DoTestArray(vbtest, ((), ()))
    _DoTestArray(vbtest, tuple(range(1, 100)))
    _DoTestArray(vbtest, (1.0, 2.0, 3.0))
    _DoTestArray(vbtest, tuple('Hello from Python'.split()))
    _DoTestArray(vbtest, (vbtest, vbtest))
    _DoTestArray(vbtest, (1, 2.0, '3'))
    _DoTestArray(vbtest, (1, (vbtest, vbtest), ('3', '4')))
    _DoTestArray(vbtest, ((1, 2, 3), (4, 5, 6)))
    _DoTestArray(vbtest, ((vbtest, vbtest, vbtest), (vbtest, vbtest, vbtest)))
    arrayData = (((1, 2), (3, 4), (5, 6)), ((7, 8), (9, 10), (11, 12)))
    arrayData = (((vbtest, vbtest), (vbtest, vbtest), (vbtest, vbtest)), ((vbtest, vbtest), (vbtest, vbtest), (vbtest, vbtest)))
    _DoTestArray(vbtest, arrayData)
    _DoTestArray(vbtest, (vbtest, 2.0, '3'))
    _DoTestArray(vbtest, (1, 2.0, vbtest))
    expected_exception = None
    arrayData = (((1, 2, 1), (3, 4), (5, 6)), ((7, 8), (9, 10), (11, 12)))
    _DoTestArray(vbtest, arrayData, expected_exception)
    arrayData = (((vbtest, vbtest),), ((vbtest,),))
    _DoTestArray(vbtest, arrayData, expected_exception)
    arrayData = (((1, 2), (3, 4), (5, 6, 8)), ((7, 8), (9, 10), (11, 12)))
    _DoTestArray(vbtest, arrayData, expected_exception)
    callback_ob = wrap(TestObject(), useDispatcher=useDispatcher)
    print("** Expecting a 'ValueError' exception to be printed next:")
    try:
        vbtest.DoCallbackSafeArraySizeFail(callback_ob)
    except pythoncom.com_error as exc:
        assert exc.excepinfo[1] == 'Python COM Server Internal Error', f"Didnt get the correct exception - '{exc}'"
    if bUseGenerated:
        testData = 'Mark was here'.split()
        (resultData, byRefParam) = vbtest.PassSAFEARRAY(testData)
        if testData != list(resultData):
            raise error('The safe array data was not what we expected - got ' + str(resultData))
        if testData != list(byRefParam):
            raise error('The safe array data was not what we expected - got ' + str(byRefParam))
        testData = [1.0, 2.0, 3.0]
        (resultData, byRefParam) = vbtest.PassSAFEARRAYVariant(testData)
        assert testData == list(byRefParam)
        assert testData == list(resultData)
        testData = ['hi', 'from', 'Python']
        (resultData, byRefParam) = vbtest.PassSAFEARRAYVariant(testData)
        assert testData == list(byRefParam), "Expected '{}', got '{}'".format(testData, list(byRefParam))
        assert testData == list(resultData), "Expected '{}', got '{}'".format(testData, list(resultData))
        testData = [1, 2.0, '3']
        (resultData, byRefParam) = vbtest.PassSAFEARRAYVariant(testData)
        assert testData == list(byRefParam)
        assert testData == list(resultData)
    print('Array tests passed')

def TestStructs(vbtest):
    if False:
        for i in range(10):
            print('nop')
    try:
        vbtest.IntProperty = 'One'
        raise error('Should have failed by now')
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_TYPEMISMATCH:
            raise error('Expected DISP_E_TYPEMISMATCH')
    s = vbtest.StructProperty
    if s.int_val != 99 or str(s.str_val) != 'hello':
        raise error('The struct value was not correct')
    s.str_val = 'Hi from Python'
    s.int_val = 11
    if s.int_val != 11 or str(s.str_val) != 'Hi from Python':
        raise error('The struct value didnt persist!')
    if s.sub_val.int_val != 66 or str(s.sub_val.str_val) != 'sub hello':
        raise error('The sub-struct value was not correct')
    sub = s.sub_val
    sub.int_val = 22
    if sub.int_val != 22:
        print(sub.int_val)
        raise error('The sub-struct value didnt persist!')
    if s.sub_val.int_val != 22:
        print(s.sub_val.int_val)
        raise error('The sub-struct value (re-fetched) didnt persist!')
    if s.sub_val.array_val[0].int_val != 0 or str(s.sub_val.array_val[0].str_val) != 'zero':
        print(s.sub_val.array_val[0].int_val)
        raise error('The array element wasnt correct')
    s.sub_val.array_val[0].int_val = 99
    s.sub_val.array_val[1].int_val = 66
    if s.sub_val.array_val[0].int_val != 99 or s.sub_val.array_val[1].int_val != 66:
        print(s.sub_val.array_val[0].int_val)
        raise error('The array element didnt persist.')
    vbtest.StructProperty = s
    s = vbtest.StructProperty
    if s.int_val != 11 or str(s.str_val) != 'Hi from Python':
        raise error('After sending to VB, the struct value didnt persist!')
    if s.sub_val.array_val[0].int_val != 99:
        raise error('After sending to VB, the struct array value didnt persist!')
    assert s == s
    assert s is not None
    try:
        s < None
        raise error('Expected type error')
    except TypeError:
        pass
    try:
        None < s
        raise error('Expected type error')
    except TypeError:
        pass
    assert s != s.sub_val
    import copy
    s2 = copy.copy(s)
    assert s is not s2
    assert s == s2
    s2.int_val = 123
    assert s != s2
    s2 = vbtest.GetStructFunc()
    assert s == s2
    vbtest.SetStructSub(s2)
    s = win32com.client.Record('VBStruct', vbtest)
    assert s.int_val == 0, 'new struct inst initialized correctly!'
    s.int_val = -1
    vbtest.SetStructSub(s)
    assert vbtest.GetStructFunc().int_val == -1, 'new struct didnt make the round trip!'
    s_array = vbtest.StructArrayProperty
    assert s_array is None, 'Expected None from the uninitialized VB array'
    vbtest.MakeStructArrayProperty(3)
    s_array = vbtest.StructArrayProperty
    assert len(s_array) == 3
    for i in range(len(s_array)):
        assert s_array[i].int_val == i
        assert s_array[i].sub_val.int_val == i
        assert s_array[i].sub_val.array_val[0].int_val == i
        assert s_array[i].sub_val.array_val[1].int_val == i + 1
        assert s_array[i].sub_val.array_val[2].int_val == i + 2
    try:
        s.bad_attribute
        raise RuntimeError('Could get a bad attribute')
    except AttributeError:
        pass
    m = s.__members__
    assert m[0] == 'int_val' and m[1] == 'str_val' and (m[2] == 'ob_val') and (m[3] == 'sub_val'), m
    try:
        s.foo
        raise RuntimeError('Expected attribute error')
    except AttributeError as exc:
        assert 'foo' in str(exc), exc
    expected = 'com_struct(int_val={!r}, str_val={!r}, ob_val={!r}, sub_val={!r})'.format(s.int_val, s.str_val, s.ob_val, s.sub_val)
    if repr(s) != expected:
        print('Expected repr:', expected)
        print('Actual repr  :', repr(s))
        raise RuntimeError('repr() of record object failed')
    print('Struct/Record tests passed')

def TestVBInterface(ob):
    if False:
        return 10
    t = ob.GetInterfaceTester(2)
    if t.getn() != 2:
        raise error('Initial value wrong')
    t.setn(3)
    if t.getn() != 3:
        raise error('New value wrong')

def TestObjectSemantics(ob):
    if False:
        i = 10
        return i + 15
    assert ob == ob._oleobj_
    assert not ob != ob._oleobj_
    assert ob._oleobj_ == ob
    assert not ob._oleobj_ != ob
    assert ob._oleobj_ == ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown)
    assert not ob._oleobj_ != ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown)
    assert ob._oleobj_ is not None
    assert None != ob._oleobj_
    assert ob is not None
    assert None != ob
    try:
        ob < None
        raise error('Expected type error')
    except TypeError:
        pass
    try:
        None < ob
        raise error('Expected type error')
    except TypeError:
        pass
    assert ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown) == ob._oleobj_
    assert not ob._oleobj_.QueryInterface(pythoncom.IID_IUnknown) != ob._oleobj_
    assert ob._oleobj_ == ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch)
    assert not ob._oleobj_ != ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch)
    assert ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch) == ob._oleobj_
    assert not ob._oleobj_.QueryInterface(pythoncom.IID_IDispatch) != ob._oleobj_
    print('Object semantic tests passed')

def DoTestAll():
    if False:
        return 10
    o = win32com.client.Dispatch('PyCOMVBTest.Tester')
    TestObjectSemantics(o)
    TestVB(o, 1)
    o = win32com.client.dynamic.DumbDispatch('PyCOMVBTest.Tester')
    TestObjectSemantics(o)
    TestVB(o, 0)

def TestAll():
    if False:
        print('Hello World!')
    win32com.client.gencache.EnsureDispatch('PyCOMVBTest.Tester')
    if not __debug__:
        raise RuntimeError('This must be run in debug mode - we use assert!')
    try:
        DoTestAll()
        print('All tests appear to have worked!')
    except:
        print('TestAll() failed!!')
        traceback.print_exc()
        raise

def suite():
    if False:
        return 10
    import unittest
    test = util.CapturingFunctionTestCase(TestAll, description='VB tests')
    suite = unittest.TestSuite()
    suite.addTest(test)
    return suite
if __name__ == '__main__':
    util.testmain()