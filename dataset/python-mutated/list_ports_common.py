from __future__ import absolute_import
import re
import glob
import os
import os.path

def numsplit(text):
    if False:
        while True:
            i = 10
    '    Convert string into a list of texts and numbers in order to support a\n    natural sorting.\n    '
    result = []
    for group in re.split('(\\d+)', text):
        if group:
            try:
                group = int(group)
            except ValueError:
                pass
            result.append(group)
    return result

class ListPortInfo(object):
    """Info collection base class for serial ports"""

    def __init__(self, device, skip_link_detection=False):
        if False:
            while True:
                i = 10
        self.device = device
        self.name = os.path.basename(device)
        self.description = 'n/a'
        self.hwid = 'n/a'
        self.vid = None
        self.pid = None
        self.serial_number = None
        self.location = None
        self.manufacturer = None
        self.product = None
        self.interface = None
        if not skip_link_detection and device is not None and os.path.islink(device):
            self.hwid = 'LINK={}'.format(os.path.realpath(device))

    def usb_description(self):
        if False:
            while True:
                i = 10
        'return a short string to name the port based on USB info'
        if self.interface is not None:
            return '{} - {}'.format(self.product, self.interface)
        elif self.product is not None:
            return self.product
        else:
            return self.name

    def usb_info(self):
        if False:
            while True:
                i = 10
        'return a string with USB related information about device'
        return 'USB VID:PID={:04X}:{:04X}{}{}'.format(self.vid or 0, self.pid or 0, ' SER={}'.format(self.serial_number) if self.serial_number is not None else '', ' LOCATION={}'.format(self.location) if self.location is not None else '')

    def apply_usb_info(self):
        if False:
            i = 10
            return i + 15
        'update description and hwid from USB data'
        self.description = self.usb_description()
        self.hwid = self.usb_info()

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, ListPortInfo) and self.device == other.device

    def __hash__(self):
        if False:
            return 10
        return hash(self.device)

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, ListPortInfo):
            raise TypeError('unorderable types: {}() and {}()'.format(type(self).__name__, type(other).__name__))
        return numsplit(self.device) < numsplit(other.device)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '{} - {}'.format(self.device, self.description)

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        'Item access: backwards compatible -> (port, desc, hwid)'
        if index == 0:
            return self.device
        elif index == 1:
            return self.description
        elif index == 2:
            return self.hwid
        else:
            raise IndexError('{} > 2'.format(index))

def list_links(devices):
    if False:
        print('Hello World!')
    '    search all /dev devices and look for symlinks to known ports already\n    listed in devices.\n    '
    links = []
    for device in glob.glob('/dev/*'):
        if os.path.islink(device) and os.path.realpath(device) in devices:
            links.append(device)
    return links
if __name__ == '__main__':
    print(ListPortInfo('dummy'))