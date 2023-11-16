import sys
sys.coinit_flags = 0
import datetime
import decimal
import os
import time
import pythoncom
import pywintypes
import win32api
import win32com
import win32com.client.connect
import win32timezone
import winerror
from win32com.client import VARIANT, CastTo, DispatchBaseClass, constants
from win32com.test.util import RegisterPythonServer
importMsg = '**** PyCOMTest is not installed ***\n  PyCOMTest is a Python test specific COM client and server.\n  It is likely this server is not installed on this machine\n  To install the server, you must get the win32com sources\n  and build it using MS Visual C++'
error = Exception
RegisterPythonServer(os.path.join(os.path.dirname(__file__), '..', 'servers', 'test_pycomtest.py'), 'Python.Test.PyCOMTest')
from win32com.client import gencache
try:
    gencache.EnsureModule('{6BCDCB60-5605-11D0-AE5F-CADD4C000000}', 0, 1, 1)
except pythoncom.com_error:
    print('The PyCOMTest module can not be located or generated.')
    print(importMsg)
    raise RuntimeError(importMsg)
from win32com import universal
universal.RegisterInterfaces('{6BCDCB60-5605-11D0-AE5F-CADD4C000000}', 0, 1, 1)
verbose = 0

def check_get_set(func, arg):
    if False:
        while True:
            i = 10
    got = func(arg)
    if got != arg:
        raise error(f'{func} failed - expected {arg!r}, got {got!r}')

def check_get_set_raises(exc, func, arg):
    if False:
        for i in range(10):
            print('nop')
    try:
        got = func(arg)
    except exc as e:
        pass
    else:
        raise error(f"{func} with arg {arg!r} didn't raise {exc} - returned {got!r}")

def progress(*args):
    if False:
        for i in range(10):
            print('nop')
    if verbose:
        for arg in args:
            print(arg, end=' ')
        print()

def TestApplyResult(fn, args, result):
    if False:
        return 10
    try:
        fnName = str(fn).split()[1]
    except:
        fnName = str(fn)
    progress('Testing ', fnName)
    pref = 'function ' + fnName
    rc = fn(*args)
    if rc != result:
        raise error(f'{pref} failed - result not {result!r} but {rc!r}')

def TestConstant(constName, pyConst):
    if False:
        return 10
    try:
        comConst = getattr(constants, constName)
    except:
        raise error(f'Constant {constName} missing')
    if comConst != pyConst:
        raise error(f'Constant value wrong for {constName} - got {comConst}, wanted {pyConst}')

class RandomEventHandler:

    def _Init(self):
        if False:
            return 10
        self.fireds = {}

    def OnFire(self, no):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.fireds[no] = self.fireds[no] + 1
        except KeyError:
            self.fireds[no] = 0

    def OnFireWithNamedParams(self, no, a_bool, out1, out2):
        if False:
            while True:
                i = 10
        Missing = pythoncom.Missing
        if no is not Missing:
            assert no in self.fireds
            assert no + 1 == out1, "expecting 'out1' param to be ID+1"
            assert no + 2 == out2, "expecting 'out2' param to be ID+2"
        assert a_bool is Missing or isinstance(a_bool, bool), 'middle param not a bool'
        return (out1 + 2, out2 + 2)

    def _DumpFireds(self):
        if False:
            print('Hello World!')
        if not self.fireds:
            print('ERROR: Nothing was received!')
        for (firedId, no) in self.fireds.items():
            progress('ID %d fired %d times' % (firedId, no))

def TestCommon(o, is_generated):
    if False:
        for i in range(10):
            print('nop')
    progress('Getting counter')
    counter = o.GetSimpleCounter()
    TestCounter(counter, is_generated)
    progress('Checking default args')
    rc = o.TestOptionals()
    if rc[:-1] != ('def', 0, 1) or abs(rc[-1] - 3.14) > 0.01:
        print(rc)
        raise error('Did not get the optional values correctly')
    rc = o.TestOptionals('Hi', 2, 3, 1.1)
    if rc[:-1] != ('Hi', 2, 3) or abs(rc[-1] - 1.1) > 0.01:
        print(rc)
        raise error('Did not get the specified optional values correctly')
    rc = o.TestOptionals2(0)
    if rc != (0, '', 1):
        print(rc)
        raise error('Did not get the optional2 values correctly')
    rc = o.TestOptionals2(1.1, 'Hi', 2)
    if rc[1:] != ('Hi', 2) or abs(rc[0] - 1.1) > 0.01:
        print(rc)
        raise error('Did not get the specified optional2 values correctly')
    progress('Checking getting/passing IUnknown')
    check_get_set(o.GetSetUnknown, o)
    progress('Checking getting/passing IDispatch')
    expected_class = o.__class__
    expected_class = getattr(expected_class, 'default_interface', expected_class)
    if not isinstance(o.GetSetDispatch(o), expected_class):
        raise error(f'GetSetDispatch failed: {o.GetSetDispatch(o)!r}')
    progress('Checking getting/passing IDispatch of known type')
    expected_class = o.__class__
    expected_class = getattr(expected_class, 'default_interface', expected_class)
    if o.GetSetInterface(o).__class__ != expected_class:
        raise error('GetSetDispatch failed')
    progress('Checking misc args')
    check_get_set(o.GetSetVariant, 4)
    check_get_set(o.GetSetVariant, 'foo')
    check_get_set(o.GetSetVariant, o)
    check_get_set(o.GetSetInt, 0)
    check_get_set(o.GetSetInt, -1)
    check_get_set(o.GetSetInt, 1)
    check_get_set(o.GetSetUnsignedInt, 0)
    check_get_set(o.GetSetUnsignedInt, 1)
    check_get_set(o.GetSetUnsignedInt, 2147483648)
    if o.GetSetUnsignedInt(-1) != 4294967295:
        raise error('unsigned -1 failed')
    check_get_set(o.GetSetLong, 0)
    check_get_set(o.GetSetLong, -1)
    check_get_set(o.GetSetLong, 1)
    check_get_set(o.GetSetUnsignedLong, 0)
    check_get_set(o.GetSetUnsignedLong, 1)
    check_get_set(o.GetSetUnsignedLong, 2147483648)
    if o.GetSetUnsignedLong(-1) != 4294967295:
        raise error('unsigned -1 failed')
    big = 2147483647
    for l in (big, big + 1, 1 << 65):
        check_get_set(o.GetSetVariant, l)
    progress('Checking structs')
    r = o.GetStruct()
    assert r.int_value == 99 and str(r.str_value) == 'Hello from C++'
    assert o.DoubleString('foo') == 'foofoo'
    progress('Checking var args')
    o.SetVarArgs('Hi', 'There', 'From', 'Python', 1)
    if o.GetLastVarArgs() != ('Hi', 'There', 'From', 'Python', 1):
        raise error('VarArgs failed -' + str(o.GetLastVarArgs()))
    progress('Checking arrays')
    l = []
    TestApplyResult(o.SetVariantSafeArray, (l,), len(l))
    l = [1, 2, 3, 4]
    TestApplyResult(o.SetVariantSafeArray, (l,), len(l))
    TestApplyResult(o.CheckVariantSafeArray, ((1, 2, 3, 4),), 1)
    TestApplyResult(o.SetBinSafeArray, (memoryview(b'foo\x00bar'),), 7)
    progress('Checking properties')
    o.LongProp = 3
    if o.LongProp != 3 or o.IntProp != 3:
        raise error('Property value wrong - got %d/%d' % (o.LongProp, o.IntProp))
    o.LongProp = o.IntProp = -3
    if o.LongProp != -3 or o.IntProp != -3:
        raise error('Property value wrong - got %d/%d' % (o.LongProp, o.IntProp))
    check = 3 * 10 ** 9
    o.ULongProp = check
    if o.ULongProp != check:
        raise error('Property value wrong - got %d (expected %d)' % (o.ULongProp, check))
    TestApplyResult(o.Test, ('Unused', 99), 1)
    TestApplyResult(o.Test, ('Unused', -1), 1)
    TestApplyResult(o.Test, ('Unused', 1 == 1), 1)
    TestApplyResult(o.Test, ('Unused', 0), 0)
    TestApplyResult(o.Test, ('Unused', 1 == 0), 0)
    assert o.DoubleString('foo') == 'foofoo'
    TestConstant('ULongTest1', 4294967295)
    TestConstant('ULongTest2', 2147483647)
    TestConstant('LongTest1', -2147483647)
    TestConstant('LongTest2', 2147483647)
    TestConstant('UCharTest', 255)
    TestConstant('CharTest', -1)
    TestConstant('StringTest', 'Hello Wo®ld')
    progress('Checking dates and times')
    now = win32timezone.now()
    now = now.replace(microsecond=0)
    later = now + datetime.timedelta(seconds=1)
    TestApplyResult(o.EarliestDate, (now, later), now)
    assert o.MakeDate(18712.308206013888) == datetime.datetime.fromisoformat('1951-03-25 07:23:49+00:00')
    progress('Checking currency')
    pythoncom.__future_currency__ = 1
    if o.CurrencyProp != 0:
        raise error(f'Expecting 0, got {o.CurrencyProp!r}')
    for val in ('1234.5678', '1234.56', '1234'):
        o.CurrencyProp = decimal.Decimal(val)
        if o.CurrencyProp != decimal.Decimal(val):
            raise error(f'{val} got {o.CurrencyProp!r}')
    v1 = decimal.Decimal('1234.5678')
    TestApplyResult(o.DoubleCurrency, (v1,), v1 * 2)
    v2 = decimal.Decimal('9012.3456')
    TestApplyResult(o.AddCurrencies, (v1, v2), v1 + v2)
    TestTrickyTypesWithVariants(o, is_generated)
    progress('Checking win32com.client.VARIANT')
    TestPyVariant(o, is_generated)

def TestTrickyTypesWithVariants(o, is_generated):
    if False:
        i = 10
        return i + 15
    if is_generated:
        got = o.TestByRefVariant(2)
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_VARIANT, 2)
        o.TestByRefVariant(v)
        got = v.value
    if got != 4:
        raise error('TestByRefVariant failed')
    if is_generated:
        got = o.TestByRefString('Foo')
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_BSTR, 'Foo')
        o.TestByRefString(v)
        got = v.value
    if got != 'FooFoo':
        raise error('TestByRefString failed')
    vals = [1, 2, 3, 4]
    if is_generated:
        arg = vals
    else:
        arg = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_UI1, vals)
    TestApplyResult(o.SetBinSafeArray, (arg,), len(vals))
    vals = [0, 1.1, 2.2, 3.3]
    if is_generated:
        arg = vals
    else:
        arg = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, vals)
    TestApplyResult(o.SetDoubleSafeArray, (arg,), len(vals))
    if is_generated:
        arg = vals
    else:
        arg = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R4, vals)
    TestApplyResult(o.SetFloatSafeArray, (arg,), len(vals))
    vals = [1.1, 2.2, 3.3, 4.4]
    expected = (1.1 * 2, 2.2 * 2, 3.3 * 2, 4.4 * 2)
    if is_generated:
        TestApplyResult(o.ChangeDoubleSafeArray, (vals,), expected)
    else:
        arg = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_ARRAY | pythoncom.VT_R8, vals)
        o.ChangeDoubleSafeArray(arg)
        if arg.value != expected:
            raise error('ChangeDoubleSafeArray got the wrong value')
    if is_generated:
        got = o.DoubleInOutString('foo')
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_BSTR, 'foo')
        o.DoubleInOutString(v)
        got = v.value
    assert got == 'foofoo', got
    val = decimal.Decimal('1234.5678')
    if is_generated:
        got = o.DoubleCurrencyByVal(val)
    else:
        v = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_CY, val)
        o.DoubleCurrencyByVal(v)
        got = v.value
    assert got == val * 2

def TestDynamic():
    if False:
        i = 10
        return i + 15
    progress('Testing Dynamic')
    import win32com.client.dynamic
    o = win32com.client.dynamic.DumbDispatch('PyCOMTest.PyCOMTest')
    TestCommon(o, False)
    counter = win32com.client.dynamic.DumbDispatch('PyCOMTest.SimpleCounter')
    TestCounter(counter, False)
    try:
        check_get_set_raises(ValueError, o.GetSetInt, 'foo')
        raise error('no exception raised')
    except pythoncom.com_error as exc:
        if exc.hresult != winerror.DISP_E_TYPEMISMATCH:
            raise
    arg1 = VARIANT(pythoncom.VT_R4 | pythoncom.VT_BYREF, 2.0)
    arg2 = VARIANT(pythoncom.VT_BOOL | pythoncom.VT_BYREF, True)
    arg3 = VARIANT(pythoncom.VT_I4 | pythoncom.VT_BYREF, 4)
    o.TestInOut(arg1, arg2, arg3)
    assert arg1.value == 4.0, arg1
    assert arg2.value == False
    assert arg3.value == 8

def TestGenerated():
    if False:
        while True:
            i = 10
    from win32com.client.gencache import EnsureDispatch
    o = EnsureDispatch('PyCOMTest.PyCOMTest')
    TestCommon(o, True)
    counter = EnsureDispatch('PyCOMTest.SimpleCounter')
    TestCounter(counter, True)
    from win32com.client.CLSIDToClass import GetClass
    coclass_o = GetClass('{8EE0C520-5605-11D0-AE5F-CADD4C000000}')()
    TestCommon(coclass_o, True)
    assert bool(coclass_o)
    coclass = GetClass('{B88DD310-BAE8-11D0-AE86-76F2C1000000}')()
    TestCounter(coclass, True)
    (i1, i2) = o.GetMultipleInterfaces()
    if not isinstance(i1, DispatchBaseClass) or not isinstance(i2, DispatchBaseClass):
        raise error(f"GetMultipleInterfaces did not return instances - got '{i1}', '{i2}'")
    del i1
    del i2
    check_get_set_raises(OverflowError, o.GetSetInt, 2147483648)
    check_get_set_raises(OverflowError, o.GetSetLong, 2147483648)
    check_get_set_raises(ValueError, o.GetSetInt, 'foo')
    check_get_set_raises(ValueError, o.GetSetLong, 'foo')
    try:
        o.SetVariantSafeArray('foo')
        raise error('Expected a type error')
    except TypeError:
        pass
    try:
        o.SetVariantSafeArray(666)
        raise error('Expected a type error')
    except TypeError:
        pass
    o.GetSimpleSafeArray(None)
    TestApplyResult(o.GetSimpleSafeArray, (None,), tuple(range(10)))
    resultCheck = (tuple(range(5)), tuple(range(10)), tuple(range(20)))
    TestApplyResult(o.GetSafeArrays, (None, None, None), resultCheck)
    l = []
    TestApplyResult(o.SetIntSafeArray, (l,), len(l))
    l = [1, 2, 3, 4]
    TestApplyResult(o.SetIntSafeArray, (l,), len(l))
    ll = [1, 2, 3, 4294967296]
    TestApplyResult(o.SetLongLongSafeArray, (ll,), len(ll))
    TestApplyResult(o.SetULongLongSafeArray, (ll,), len(ll))
    TestApplyResult(o.Test2, (constants.Attr2,), constants.Attr2)
    TestApplyResult(o.Test3, (constants.Attr2,), constants.Attr2)
    TestApplyResult(o.Test4, (constants.Attr2,), constants.Attr2)
    TestApplyResult(o.Test5, (constants.Attr2,), constants.Attr2)
    TestApplyResult(o.Test6, (constants.WideAttr1,), constants.WideAttr1)
    TestApplyResult(o.Test6, (constants.WideAttr2,), constants.WideAttr2)
    TestApplyResult(o.Test6, (constants.WideAttr3,), constants.WideAttr3)
    TestApplyResult(o.Test6, (constants.WideAttr4,), constants.WideAttr4)
    TestApplyResult(o.Test6, (constants.WideAttr5,), constants.WideAttr5)
    TestApplyResult(o.TestInOut, (2.0, True, 4), (4.0, False, 8))
    o.SetParamProp(0, 1)
    if o.ParamProp(0) != 1:
        raise RuntimeError(o.paramProp(0))
    o2 = CastTo(o, 'IPyCOMTest')
    if o != o2:
        raise error('CastTo should have returned the same object')
    progress('Testing connection points')
    o2 = win32com.client.DispatchWithEvents(o, RandomEventHandler)
    TestEvents(o2, o2)
    handler = win32com.client.WithEvents(o, RandomEventHandler)
    TestEvents(o, handler)
    progress('Finished generated .py test.')

def TestEvents(o, handler):
    if False:
        for i in range(10):
            print('nop')
    sessions = []
    handler._Init()
    try:
        for i in range(3):
            session = o.Start()
            sessions.append(session)
        time.sleep(0.5)
    finally:
        for session in sessions:
            o.Stop(session)
        handler._DumpFireds()
        handler.close()

def _TestPyVariant(o, is_generated, val, checker=None):
    if False:
        for i in range(10):
            print('nop')
    if is_generated:
        (vt, got) = o.GetVariantAndType(val)
    else:
        var_vt = VARIANT(pythoncom.VT_UI2 | pythoncom.VT_BYREF, 0)
        var_result = VARIANT(pythoncom.VT_VARIANT | pythoncom.VT_BYREF, 0)
        o.GetVariantAndType(val, var_vt, var_result)
        vt = var_vt.value
        got = var_result.value
    if checker is not None:
        checker(got)
        return
    assert vt == val.varianttype, (vt, val.varianttype)
    if isinstance(val.value, (tuple, list)):
        check = [v.value if isinstance(v, VARIANT) else v for v in val.value]
        got = list(got)
    else:
        check = val.value
    assert type(check) == type(got), (type(check), type(got))
    assert check == got, (check, got)

def _TestPyVariantFails(o, is_generated, val, exc):
    if False:
        print('Hello World!')
    try:
        _TestPyVariant(o, is_generated, val)
        raise error(f"Setting {val!r} didn't raise {exc}")
    except exc:
        pass

def TestPyVariant(o, is_generated):
    if False:
        print('Hello World!')
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_UI1, 1))
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_UI4, [1, 2, 3]))
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_BSTR, 'hello'))
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_BSTR, ['hello', 'there']))

    def check_dispatch(got):
        if False:
            while True:
                i = 10
        assert isinstance(got._oleobj_, pythoncom.TypeIIDs[pythoncom.IID_IDispatch])
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_DISPATCH, o), check_dispatch)
    _TestPyVariant(o, is_generated, VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, [o]))
    v = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_VARIANT, [VARIANT(pythoncom.VT_UI4, 1), VARIANT(pythoncom.VT_UI4, 2), VARIANT(pythoncom.VT_UI4, 3)])
    _TestPyVariant(o, is_generated, v)
    _TestPyVariantFails(o, is_generated, VARIANT(pythoncom.VT_UI1, 'foo'), ValueError)

def TestCounter(counter, bIsGenerated):
    if False:
        i = 10
        return i + 15
    progress('Testing counter', repr(counter))
    import random
    for i in range(50):
        num = int(random.random() * len(counter))
        try:
            if bIsGenerated:
                ret = counter.Item(num + 1)
            else:
                ret = counter[num]
            if ret != num + 1:
                raise error('Random access into element %d failed - return was %s' % (num, repr(ret)))
        except IndexError:
            raise error('** IndexError accessing collection element %d' % num)
    num = 0
    if bIsGenerated:
        counter.SetTestProperty(1)
        counter.TestProperty = 1
        counter.SetTestProperty(1, 2)
        if counter.TestPropertyWithDef != 0:
            raise error('Unexpected property set value!')
        if counter.TestPropertyNoDef(1) != 1:
            raise error('Unexpected property set value!')
    else:
        pass
    counter.LBound = 1
    counter.UBound = 10
    if counter.LBound != 1 or counter.UBound != 10:
        print('** Error - counter did not keep its properties')
    if bIsGenerated:
        bounds = counter.GetBounds()
        if bounds[0] != 1 or bounds[1] != 10:
            raise error('** Error - counter did not give the same properties back')
        counter.SetBounds(bounds[0], bounds[1])
    for item in counter:
        num = num + 1
    if num != len(counter):
        raise error('*** Length of counter and loop iterations dont match ***')
    if num != 10:
        raise error('*** Unexpected number of loop iterations ***')
    try:
        counter = iter(counter)._iter_.Clone()
    except AttributeError:
        progress('Finished testing counter (but skipped the iterator stuff')
        return
    counter.Reset()
    num = 0
    for item in counter:
        num = num + 1
    if num != 10:
        raise error('*** Unexpected number of loop iterations - got %d ***' % num)
    progress('Finished testing counter')

def TestLocalVTable(ob):
    if False:
        print('Hello World!')
    if ob.DoubleString('foo') != 'foofoo':
        raise error("couldn't foofoo")

def TestVTable(clsctx=pythoncom.CLSCTX_ALL):
    if False:
        print('Hello World!')
    ob = win32com.client.Dispatch('Python.Test.PyCOMTest')
    TestLocalVTable(ob)
    tester = win32com.client.Dispatch('PyCOMTest.PyCOMTest')
    testee = pythoncom.CoCreateInstance('Python.Test.PyCOMTest', None, clsctx, pythoncom.IID_IUnknown)
    try:
        tester.TestMyInterface(None)
    except pythoncom.com_error as details:
        pass
    tester.TestMyInterface(testee)

def TestVTable2():
    if False:
        return 10
    ob = win32com.client.Dispatch('Python.Test.PyCOMTest')
    iid = pythoncom.InterfaceNames['IPyCOMTest']
    clsid = 'Python.Test.PyCOMTest'
    clsctx = pythoncom.CLSCTX_SERVER
    try:
        testee = pythoncom.CoCreateInstance(clsid, None, clsctx, iid)
    except TypeError:
        pass

def TestVTableMI():
    if False:
        for i in range(10):
            print('nop')
    clsctx = pythoncom.CLSCTX_SERVER
    ob = pythoncom.CoCreateInstance('Python.Test.PyCOMTestMI', None, clsctx, pythoncom.IID_IUnknown)
    ob.QueryInterface(pythoncom.IID_IStream)
    ob.QueryInterface(pythoncom.IID_IStorage)
    ob.QueryInterface(pythoncom.IID_IDispatch)
    iid = pythoncom.InterfaceNames['IPyCOMTest']
    try:
        ob.QueryInterface(iid)
    except TypeError:
        pass

def TestQueryInterface(long_lived_server=0, iterations=5):
    if False:
        print('Hello World!')
    tester = win32com.client.Dispatch('PyCOMTest.PyCOMTest')
    if long_lived_server:
        t0 = win32com.client.Dispatch('Python.Test.PyCOMTest', clsctx=pythoncom.CLSCTX_LOCAL_SERVER)
    prompt = ['Testing QueryInterface without long-lived local-server #%d of %d...', 'Testing QueryInterface with long-lived local-server #%d of %d...']
    for i in range(iterations):
        progress(prompt[long_lived_server != 0] % (i + 1, iterations))
        tester.TestQueryInterface()

class Tester(win32com.test.util.TestCase):

    def testVTableInProc(self):
        if False:
            print('Hello World!')
        for i in range(3):
            progress('Testing VTables in-process #%d...' % (i + 1))
            TestVTable(pythoncom.CLSCTX_INPROC_SERVER)

    def testVTableLocalServer(self):
        if False:
            print('Hello World!')
        for i in range(3):
            progress('Testing VTables out-of-process #%d...' % (i + 1))
            TestVTable(pythoncom.CLSCTX_LOCAL_SERVER)

    def testVTable2(self):
        if False:
            i = 10
            return i + 15
        for i in range(3):
            TestVTable2()

    def testVTableMI(self):
        if False:
            return 10
        for i in range(3):
            TestVTableMI()

    def testMultiQueryInterface(self):
        if False:
            return 10
        TestQueryInterface(0, 6)
        TestQueryInterface(1, 6)

    def testDynamic(self):
        if False:
            for i in range(10):
                print('nop')
        TestDynamic()

    def testGenerated(self):
        if False:
            for i in range(10):
                print('nop')
        TestGenerated()
if __name__ == '__main__':

    def NullThreadFunc():
        if False:
            while True:
                i = 10
        pass
    import _thread
    _thread.start_new(NullThreadFunc, ())
    if '-v' in sys.argv:
        verbose = 1
    win32com.test.util.testmain()