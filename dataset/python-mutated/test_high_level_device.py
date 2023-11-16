import unittest
import openrazer_daemon.device
DEVICE1_SERIAL = 'XX000000'
DEVICE1_ID = '0000:0000:0000.0000'
DEVICE2_SERIAL = 'XX000001'
DEVICE2_ID = '0000:0000:0000.0001'

class DummyDBusObject(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.notify_msg = None
        self.parent = None

    def notify(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.notify_msg = msg

    def register_parent(self, parent):
        if False:
            i = 10
            return i + 15
        self.parent = parent

    def notify_parent(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.parent.notify_parent(msg)

class DummyParentObject(object):

    def __init__(self):
        if False:
            return 10
        self.notify_msg = None
        self.notify_device = None

    def notify(self, device_object, msg):
        if False:
            print('Hello World!')
        self.notify_device = device_object
        self.notify_msg = msg

class DeviceTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        pass

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    def test_device_properties(self):
        if False:
            for i in range(10):
                print('nop')
        dbus_object = DummyDBusObject()
        device_object = openrazer_daemon.device.Device(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        self.assertEqual(device_object.device_id, DEVICE1_ID)
        self.assertEqual(device_object.serial, DEVICE1_SERIAL)
        self.assertEqual(device_object.dbus, dbus_object)

    def test_device_register_parent(self):
        if False:
            while True:
                i = 10
        dbus_object = DummyDBusObject()
        parent_object = DummyParentObject()
        device_object = openrazer_daemon.device.Device(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        device_object.register_parent(parent_object)
        self.assertEqual(device_object._parent, parent_object)

    def test_device_notify_child(self):
        if False:
            print('Hello World!')
        msg = ('test', 1)
        dbus_object = DummyDBusObject()
        device_object = openrazer_daemon.device.Device(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        device_object.notify_child(msg)
        self.assertEqual(dbus_object.notify_msg, msg)

    def test_device_notify_parent(self):
        if False:
            print('Hello World!')
        msg = ('test', 1)
        dbus_object = DummyDBusObject()
        parent_object = DummyParentObject()
        device_object = openrazer_daemon.device.Device(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        device_object.register_parent(parent_object)
        device_object.notify_parent(msg)
        self.assertEqual(parent_object.notify_msg, msg)
        self.assertEqual(parent_object.notify_device, device_object)

class DeviceCollectionTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.device_collection = openrazer_daemon.device.DeviceCollection()

    def test_add(self):
        if False:
            return 10
        dbus_object = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        self.assertIn(DEVICE1_ID, self.device_collection._id_map)
        self.assertIn(DEVICE1_SERIAL, self.device_collection._serial_map)
        device_obj_from_id = self.device_collection._id_map[DEVICE1_ID]
        device_obj_from_serial = self.device_collection._serial_map[DEVICE1_SERIAL]
        self.assertIs(device_obj_from_id, device_obj_from_serial)

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        dbus_object = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        device_obj_by_id = self.device_collection[DEVICE1_ID]
        device_obj_by_serial = self.device_collection[DEVICE1_SERIAL]
        self.assertIs(device_obj_by_id, device_obj_by_serial)

    def test_invalid_get(self):
        if False:
            print('Hello World!')
        try:
            device = self.device_collection.get('INVALID')
        except IndexError:
            pass

    def test_contains(self):
        if False:
            while True:
                i = 10
        dbus_object = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        self.assertIn(DEVICE1_ID, self.device_collection)
        self.assertIn(DEVICE1_SERIAL, self.device_collection)

    def test_remove(self):
        if False:
            return 10
        dbus_object = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        self.assertIn(DEVICE1_ID, self.device_collection)
        self.device_collection.remove(DEVICE1_ID)
        self.assertNotIn(DEVICE1_ID, self.device_collection)
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        self.assertIn(DEVICE1_ID, self.device_collection)
        self.device_collection.remove(DEVICE1_SERIAL)
        self.assertNotIn(DEVICE1_SERIAL, self.device_collection)

    def test_items(self):
        if False:
            return 10
        dbus_object = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        (device_id, device_obj1) = list(self.device_collection.id_items())[0]
        (device_serial, device_obj2) = list(self.device_collection.serial_items())[0]
        self.assertEqual(device_id, DEVICE1_ID)
        self.assertEqual(device_serial, DEVICE1_SERIAL)
        self.assertIs(device_obj1, device_obj2)

    def test_iter(self):
        if False:
            i = 10
            return i + 15
        dbus_object = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object)
        devices = [self.device_collection.get(DEVICE1_ID)]
        for device in self.device_collection:
            devices.remove(device)
        self.assertEqual(len(devices), 0)

    def test_serials(self):
        if False:
            for i in range(10):
                print('nop')
        dbus_object1 = DummyDBusObject()
        dbus_object2 = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object1)
        self.device_collection.add(DEVICE2_ID, DEVICE2_SERIAL, dbus_object2)
        serials = self.device_collection.serials()
        self.assertIn(DEVICE1_SERIAL, serials)
        self.assertIn(DEVICE2_SERIAL, serials)

    def test_devices(self):
        if False:
            return 10
        dbus_object1 = DummyDBusObject()
        dbus_object2 = DummyDBusObject()
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object1)
        self.device_collection.add(DEVICE2_ID, DEVICE2_SERIAL, dbus_object2)
        device_list = self.device_collection.devices
        available_dbus = [dbus_object1, dbus_object2]
        for device in device_list:
            available_dbus.remove(device.dbus)
        self.assertEqual(len(available_dbus), 0)

    def test_cross_device_notify(self):
        if False:
            while True:
                i = 10
        dbus_object1 = DummyDBusObject()
        dbus_object2 = DummyDBusObject()
        msg = ('test', 1)
        self.device_collection.add(DEVICE1_ID, DEVICE1_SERIAL, dbus_object1)
        self.device_collection.add(DEVICE2_ID, DEVICE2_SERIAL, dbus_object2)
        self.assertIs(dbus_object1.notify_msg, None)
        self.assertIs(dbus_object2.notify_msg, None)
        dbus_object1.notify_parent(msg)
        self.assertIs(dbus_object1.notify_msg, None)
        self.assertIs(dbus_object2.notify_msg, msg)