from caffe2.python import scope, core, workspace
import unittest
import threading
import time
SUCCESS_COUNT = 0

def thread_runner(idx, testobj):
    if False:
        return 10
    global SUCCESS_COUNT
    testobj.assertEquals(scope.CurrentNameScope(), '')
    testobj.assertEquals(scope.CurrentDeviceScope(), None)
    namescope = 'namescope_{}'.format(idx)
    dsc = core.DeviceOption(workspace.GpuDeviceType, idx)
    with scope.DeviceScope(dsc):
        with scope.NameScope(namescope):
            testobj.assertEquals(scope.CurrentNameScope(), namescope + '/')
            testobj.assertEquals(scope.CurrentDeviceScope(), dsc)
            time.sleep(0.01 + idx * 0.01)
            testobj.assertEquals(scope.CurrentNameScope(), namescope + '/')
            testobj.assertEquals(scope.CurrentDeviceScope(), dsc)
    testobj.assertEquals(scope.CurrentNameScope(), '')
    testobj.assertEquals(scope.CurrentDeviceScope(), None)
    SUCCESS_COUNT += 1

class TestScope(unittest.TestCase):

    def testNamescopeBasic(self):
        if False:
            while True:
                i = 10
        self.assertEqual(scope.CurrentNameScope(), '')
        with scope.NameScope('test_scope'):
            self.assertEqual(scope.CurrentNameScope(), 'test_scope/')
        self.assertEqual(scope.CurrentNameScope(), '')

    def testNamescopeAssertion(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(scope.CurrentNameScope(), '')
        try:
            with scope.NameScope('test_scope'):
                self.assertEqual(scope.CurrentNameScope(), 'test_scope/')
                raise Exception()
        except Exception:
            pass
        self.assertEqual(scope.CurrentNameScope(), '')

    def testEmptyNamescopeBasic(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(scope.CurrentNameScope(), '')
        with scope.NameScope('test_scope'):
            with scope.EmptyNameScope():
                self.assertEqual(scope.CurrentNameScope(), '')
            self.assertEqual(scope.CurrentNameScope(), 'test_scope/')

    def testDevicescopeBasic(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(scope.CurrentDeviceScope(), None)
        dsc = core.DeviceOption(workspace.GpuDeviceType, 9)
        with scope.DeviceScope(dsc):
            self.assertEqual(scope.CurrentDeviceScope(), dsc)
        self.assertEqual(scope.CurrentDeviceScope(), None)

    def testEmptyDevicescopeBasic(self):
        if False:
            return 10
        self.assertEqual(scope.CurrentDeviceScope(), None)
        dsc = core.DeviceOption(workspace.GpuDeviceType, 9)
        with scope.DeviceScope(dsc):
            self.assertEqual(scope.CurrentDeviceScope(), dsc)
            with scope.EmptyDeviceScope():
                self.assertEqual(scope.CurrentDeviceScope(), None)
            self.assertEqual(scope.CurrentDeviceScope(), dsc)
        self.assertEqual(scope.CurrentDeviceScope(), None)

    def testDevicescopeAssertion(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(scope.CurrentDeviceScope(), None)
        dsc = core.DeviceOption(workspace.GpuDeviceType, 9)
        try:
            with scope.DeviceScope(dsc):
                self.assertEqual(scope.CurrentDeviceScope(), dsc)
                raise Exception()
        except Exception:
            pass
        self.assertEqual(scope.CurrentDeviceScope(), None)

    def testTags(self):
        if False:
            print('Hello World!')
        self.assertEqual(scope.CurrentDeviceScope(), None)
        extra_info1 = ['key1:value1']
        extra_info2 = ['key2:value2']
        extra_info3 = ['key3:value3']
        extra_info_1_2 = ['key1:value1', 'key2:value2']
        extra_info_1_2_3 = ['key1:value1', 'key2:value2', 'key3:value3']
        with scope.DeviceScope(core.DeviceOption(0, extra_info=extra_info1)):
            self.assertEqual(scope.CurrentDeviceScope().extra_info, extra_info1)
            with scope.DeviceScope(core.DeviceOption(0, extra_info=extra_info2)):
                self.assertEqual(scope.CurrentDeviceScope().extra_info, extra_info_1_2)
                with scope.DeviceScope(core.DeviceOption(0, extra_info=extra_info3)):
                    self.assertEqual(scope.CurrentDeviceScope().extra_info, extra_info_1_2_3)
                self.assertEqual(scope.CurrentDeviceScope().extra_info, extra_info_1_2)
            self.assertEqual(scope.CurrentDeviceScope().extra_info, extra_info1)
        self.assertEqual(scope.CurrentDeviceScope(), None)

    def testMultiThreaded(self):
        if False:
            while True:
                i = 10
        "\n        Test that name/device scope are properly local to the thread\n        and don't interfere\n        "
        global SUCCESS_COUNT
        self.assertEqual(scope.CurrentNameScope(), '')
        self.assertEqual(scope.CurrentDeviceScope(), None)
        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=thread_runner, args=(i, self)))
        for t in threads:
            t.start()
        with scope.NameScope('master'):
            self.assertEqual(scope.CurrentDeviceScope(), None)
            self.assertEqual(scope.CurrentNameScope(), 'master/')
            for t in threads:
                t.join()
            self.assertEqual(scope.CurrentNameScope(), 'master/')
            self.assertEqual(scope.CurrentDeviceScope(), None)
        self.assertEqual(SUCCESS_COUNT, 4)