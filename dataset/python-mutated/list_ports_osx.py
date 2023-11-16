from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
iokit = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/IOKit.framework/IOKit')
cf = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
kIOMasterPortDefault = 0
kCFAllocatorDefault = ctypes.c_void_p.in_dll(cf, 'kCFAllocatorDefault')
kCFStringEncodingMacRoman = 0
kCFStringEncodingUTF8 = 134217984
kUSBVendorString = 'USB Vendor Name'
kUSBSerialNumberString = 'USB Serial Number'
io_name_size = 128
KERN_SUCCESS = 0
kern_return_t = ctypes.c_int
iokit.IOServiceMatching.restype = ctypes.c_void_p
iokit.IOServiceGetMatchingServices.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
iokit.IOServiceGetMatchingServices.restype = kern_return_t
iokit.IORegistryEntryGetParentEntry.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
iokit.IOServiceGetMatchingServices.restype = kern_return_t
iokit.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
iokit.IORegistryEntryGetPath.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
iokit.IORegistryEntryGetPath.restype = kern_return_t
iokit.IORegistryEntryGetName.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
iokit.IORegistryEntryGetName.restype = kern_return_t
iokit.IOObjectGetClass.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
iokit.IOObjectGetClass.restype = kern_return_t
iokit.IOObjectRelease.argtypes = [ctypes.c_void_p]
cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
cf.CFStringCreateWithCString.restype = ctypes.c_void_p
cf.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_uint32]
cf.CFStringGetCString.restype = ctypes.c_bool
cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
cf.CFNumberGetValue.restype = ctypes.c_void_p
cf.CFRelease.argtypes = [ctypes.c_void_p]
cf.CFRelease.restype = None
kCFNumberSInt8Type = 1
kCFNumberSInt16Type = 2
kCFNumberSInt32Type = 3
kCFNumberSInt64Type = 4

def get_string_property(device_type, property):
    if False:
        print('Hello World!')
    '\n    Search the given device for the specified string property\n\n    @param device_type Type of Device\n    @param property String to search for\n    @return Python string containing the value, or None if not found.\n    '
    key = cf.CFStringCreateWithCString(kCFAllocatorDefault, property.encode('utf-8'), kCFStringEncodingUTF8)
    CFContainer = iokit.IORegistryEntryCreateCFProperty(device_type, key, kCFAllocatorDefault, 0)
    output = None
    if CFContainer:
        output = cf.CFStringGetCStringPtr(CFContainer, 0)
        if output is not None:
            output = output.decode('utf-8')
        else:
            buffer = ctypes.create_string_buffer(io_name_size)
            success = cf.CFStringGetCString(CFContainer, ctypes.byref(buffer), io_name_size, kCFStringEncodingUTF8)
            if success:
                output = buffer.value.decode('utf-8')
        cf.CFRelease(CFContainer)
    return output

def get_int_property(device_type, property, cf_number_type):
    if False:
        i = 10
        return i + 15
    '\n    Search the given device for the specified string property\n\n    @param device_type Device to search\n    @param property String to search for\n    @param cf_number_type CFType number\n\n    @return Python string containing the value, or None if not found.\n    '
    key = cf.CFStringCreateWithCString(kCFAllocatorDefault, property.encode('utf-8'), kCFStringEncodingUTF8)
    CFContainer = iokit.IORegistryEntryCreateCFProperty(device_type, key, kCFAllocatorDefault, 0)
    if CFContainer:
        if cf_number_type == kCFNumberSInt32Type:
            number = ctypes.c_uint32()
        elif cf_number_type == kCFNumberSInt16Type:
            number = ctypes.c_uint16()
        cf.CFNumberGetValue(CFContainer, cf_number_type, ctypes.byref(number))
        cf.CFRelease(CFContainer)
        return number.value
    return None

def IORegistryEntryGetName(device):
    if False:
        return 10
    devicename = ctypes.create_string_buffer(io_name_size)
    res = iokit.IORegistryEntryGetName(device, ctypes.byref(devicename))
    if res != KERN_SUCCESS:
        return None
    return devicename.value.decode('utf-8')

def IOObjectGetClass(device):
    if False:
        i = 10
        return i + 15
    classname = ctypes.create_string_buffer(io_name_size)
    iokit.IOObjectGetClass(device, ctypes.byref(classname))
    return classname.value

def GetParentDeviceByType(device, parent_type):
    if False:
        for i in range(10):
            print('nop')
    ' Find the first parent of a device that implements the parent_type\n        @param IOService Service to inspect\n        @return Pointer to the parent type, or None if it was not found.\n    '
    parent_type = parent_type.encode('utf-8')
    while IOObjectGetClass(device) != parent_type:
        parent = ctypes.c_void_p()
        response = iokit.IORegistryEntryGetParentEntry(device, 'IOService'.encode('utf-8'), ctypes.byref(parent))
        if response != KERN_SUCCESS:
            return None
        device = parent
    return device

def GetIOServicesByType(service_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    returns iterator over specified service_type\n    '
    serial_port_iterator = ctypes.c_void_p()
    iokit.IOServiceGetMatchingServices(kIOMasterPortDefault, iokit.IOServiceMatching(service_type.encode('utf-8')), ctypes.byref(serial_port_iterator))
    services = []
    while iokit.IOIteratorIsValid(serial_port_iterator):
        service = iokit.IOIteratorNext(serial_port_iterator)
        if not service:
            break
        services.append(service)
    iokit.IOObjectRelease(serial_port_iterator)
    return services

def location_to_string(locationID):
    if False:
        while True:
            i = 10
    '\n    helper to calculate port and bus number from locationID\n    '
    loc = ['{}-'.format(locationID >> 24)]
    while locationID & 15728640:
        if len(loc) > 1:
            loc.append('.')
        loc.append('{}'.format(locationID >> 20 & 15))
        locationID <<= 4
    return ''.join(loc)

class SuitableSerialInterface(object):
    pass

def scan_interfaces():
    if False:
        while True:
            i = 10
    '\n    helper function to scan USB interfaces\n    returns a list of SuitableSerialInterface objects with name and id attributes\n    '
    interfaces = []
    for service in GetIOServicesByType('IOSerialBSDClient'):
        device = get_string_property(service, 'IOCalloutDevice')
        if device:
            usb_device = GetParentDeviceByType(service, 'IOUSBInterface')
            if usb_device:
                name = get_string_property(usb_device, 'USB Interface Name') or None
                locationID = get_int_property(usb_device, 'locationID', kCFNumberSInt32Type) or ''
                i = SuitableSerialInterface()
                i.id = locationID
                i.name = name
                interfaces.append(i)
    return interfaces

def search_for_locationID_in_interfaces(serial_interfaces, locationID):
    if False:
        while True:
            i = 10
    for interface in serial_interfaces:
        if interface.id == locationID:
            return interface.name
    return None

def comports(include_links=False):
    if False:
        return 10
    services = GetIOServicesByType('IOSerialBSDClient')
    ports = []
    serial_interfaces = scan_interfaces()
    for service in services:
        device = get_string_property(service, 'IOCalloutDevice')
        if device:
            info = list_ports_common.ListPortInfo(device)
            usb_device = GetParentDeviceByType(service, 'IOUSBHostDevice')
            if not usb_device:
                usb_device = GetParentDeviceByType(service, 'IOUSBDevice')
            if usb_device:
                info.vid = get_int_property(usb_device, 'idVendor', kCFNumberSInt16Type)
                info.pid = get_int_property(usb_device, 'idProduct', kCFNumberSInt16Type)
                info.serial_number = get_string_property(usb_device, kUSBSerialNumberString)
                info.product = IORegistryEntryGetName(usb_device) or 'n/a'
                info.manufacturer = get_string_property(usb_device, kUSBVendorString)
                locationID = get_int_property(usb_device, 'locationID', kCFNumberSInt32Type)
                info.location = location_to_string(locationID)
                info.interface = search_for_locationID_in_interfaces(serial_interfaces, locationID)
                info.apply_usb_info()
            ports.append(info)
    return ports
if __name__ == '__main__':
    for (port, desc, hwid) in sorted(comports()):
        print('{}: {} [{}]'.format(port, desc, hwid))