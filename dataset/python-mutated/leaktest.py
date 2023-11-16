from __future__ import print_function
import clr
import gc
import sys
import System
from .utils import CallableHandler, ClassMethodHandler, GenericHandler, HelloClass, StaticMethodHandler, VarCallableHandler, VariableArgsHandler, hello_func

class LeakTest(object):
    """A leak-check test for the objects implemented in the managed
       runtime. For each kind of object tested, memory should reach
       a particular level after warming up and stay essentially the
       same, net of minor fluctuation induced by GC."""

    def __init__(self):
        if False:
            return 10
        self.count = 50000
        self.quiet = 0
        self._ws = 0

    def notify(self, msg):
        if False:
            i = 10
            return i + 15
        if not self.quiet:
            print(msg)

    def start_test(self):
        if False:
            for i in range(10):
                print('nop')
        System.GC.Collect(System.GC.MaxGeneration)
        gc.collect()
        self._ws = System.Environment.WorkingSet

    def end_test(self):
        if False:
            return 10
        start = self._ws
        System.GC.Collect(System.GC.MaxGeneration)
        gc.collect()
        end = System.Environment.WorkingSet
        diff = end - start
        if diff > 0:
            diff = '+{0}'.format(diff)
        else:
            diff = '{0}'.format(diff)
        print('  start: {0}  end: {1} diff: {2}'.format(start, end, diff))
        print('')

    def run(self):
        if False:
            while True:
                i = 10
        self.testModules()
        self.testClasses()
        self.testEnumerations()
        self.testEvents()
        self.testDelegates()

    def report(self):
        if False:
            for i in range(10):
                print('nop')
        gc.collect()
        dicttype = type({})
        for item in gc.get_objects():
            if type(item) != dicttype:
                print(item, sys.getrefcount(item))

    def test_modules(self):
        if False:
            print('Hello World!')
        self.notify('Running module leak check...')
        for i in range(self.count):
            if i == 10:
                self.start_test()
            __import__('clr')
            __import__('System')
            __import__('System.IO')
            __import__('System.Net')
            __import__('System.Xml')
        self.end_test()

    def test_classes(self):
        if False:
            while True:
                i = 10
        from System.Collections import Hashtable
        from Python.Test import StringDelegate
        self.notify('Running class leak check...')
        for i in range(self.count):
            if i == 10:
                self.start_test()
            x = Hashtable()
            del x
            x = System.Int32(99)
            del x
            x = StringDelegate(hello_func)
            del x
        self.end_test()

    def test_enumerations(self):
        if False:
            for i in range(10):
                print('nop')
        import Python.Test as Test
        self.notify('Running enum leak check...')
        for i in range(self.count):
            if i == 10:
                self.start_test()
            x = Test.ByteEnum.Zero
            del x
            x = Test.SByteEnum.Zero
            del x
            x = Test.ShortEnum.Zero
            del x
            x = Test.UShortEnum.Zero
            del x
            x = Test.IntEnum.Zero
            del x
            x = Test.UIntEnum.Zero
            del x
            x = Test.LongEnum.Zero
            del x
            x = Test.ULongEnum.Zero
            del x
        self.end_test()

    def test_events(self):
        if False:
            return 10
        from Python.Test import EventTest, EventArgsTest
        self.notify('Running event leak check...')
        for i in range(self.count):
            if i == 10:
                self.start_test()
            testob = EventTest()
            handler = GenericHandler()
            testob.PublicEvent += handler.handler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= handler.handler
            del handler
            handler = VariableArgsHandler()
            testob.PublicEvent += handler.handler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= handler.handler
            del handler
            handler = CallableHandler()
            testob.PublicEvent += handler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= handler
            del handler
            handler = VarCallableHandler()
            testob.PublicEvent += handler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= handler
            del handler
            handler = StaticMethodHandler()
            StaticMethodHandler.value = None
            testob.PublicEvent += handler.handler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= handler.handler
            del handler
            handler = ClassMethodHandler()
            ClassMethodHandler.value = None
            testob.PublicEvent += handler.handler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= handler.handler
            del handler
            testob.PublicEvent += testob.GenericHandler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= testob.GenericHandler
            testob.PublicEvent += EventTest.StaticHandler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= EventTest.StaticHandler
            dict_ = {'value': None}

            def handler(sender, args, dict_=dict_):
                if False:
                    return 10
                dict_['value'] = args.value
            testob.PublicEvent += handler
            testob.PublicEvent(testob, EventArgsTest(10))
            testob.PublicEvent -= handler
            del handler
        self.end_test()

    def test_delegates(self):
        if False:
            i = 10
            return i + 15
        from Python.Test import DelegateTest, StringDelegate
        self.notify('Running delegate leak check...')
        for i in range(self.count):
            if i == 10:
                self.start_test()
            testob = DelegateTest()
            d = StringDelegate(hello_func)
            testob.CallStringDelegate(d)
            testob.stringDelegate = d
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del d
            inst = HelloClass()
            testob = DelegateTest()
            d = StringDelegate(inst.hello)
            testob.CallStringDelegate(d)
            testob.stringDelegate = d
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del inst
            del d
            testob = DelegateTest()
            d = StringDelegate(HelloClass.s_hello)
            testob.CallStringDelegate(d)
            testob.stringDelegate = d
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del d
            testob = DelegateTest()
            d = StringDelegate(HelloClass.c_hello)
            testob.CallStringDelegate(d)
            testob.stringDelegate = d
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del d
            inst = HelloClass()
            testob = DelegateTest()
            d = StringDelegate(inst)
            testob.CallStringDelegate(d)
            testob.stringDelegate = d
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del inst
            del d
            testob = DelegateTest()
            d = StringDelegate(testob.SayHello)
            testob.CallStringDelegate(d)
            testob.stringDelegate = d
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del d
            testob = DelegateTest()
            d = StringDelegate(DelegateTest.StaticSayHello)
            testob.CallStringDelegate(d)
            testob.stringDelegate = d
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del d
            testob = DelegateTest()
            d1 = StringDelegate(hello_func)
            d2 = StringDelegate(d1)
            testob.CallStringDelegate(d2)
            testob.stringDelegate = d2
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del d1
            del d2
            testob = DelegateTest()
            d1 = StringDelegate(hello_func)
            d2 = StringDelegate(hello_func)
            md = System.Delegate.Combine(d1, d2)
            testob.CallStringDelegate(md)
            testob.stringDelegate = md
            testob.stringDelegate()
            testob.stringDelegate = None
            del testob
            del d1
            del d2
            del md
        self.end_test()
if __name__ == '__main__':
    test = LeakTest()
    test.run()
    test.report()