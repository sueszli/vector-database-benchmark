"""
.. centered::
    NOTICE
    This software/technical data was produced for the U.S. Government
    under Prime Contract No. NASA-03001 and JPL Contract No. 1295026
    and is subject to FAR 52.227-14 (6/87) Rights in Data General,
    and Article GP-51, Rights in Data  General, respectively.
    This software is publicly released under MITRE case #12-3054
"""
from scapy.fields import Field, FieldLenField, LenField
from scapy.compat import raw

class SDNVValueError(Exception):

    def __init__(self, maxValue):
        if False:
            print('Hello World!')
        self.maxValue = maxValue

class SDNV:

    def __init__(self, maxValue=2 ** 32 - 1):
        if False:
            print('Hello World!')
        self.maxValue = maxValue
        return

    def setMax(self, maxValue):
        if False:
            i = 10
            return i + 15
        self.maxValue = maxValue

    def getMax(self):
        if False:
            while True:
                i = 10
        return self.maxValue

    def encode(self, number):
        if False:
            return 10
        if number > self.maxValue:
            raise SDNVValueError(self.maxValue)
        foo = bytearray()
        foo.append(number & 127)
        number = number >> 7
        while number > 0:
            thisByte = number & 127
            thisByte |= 128
            number = number >> 7
            temp = bytearray()
            temp.append(thisByte)
            foo = temp + foo
        return foo

    def decode(self, ba, offset):
        if False:
            while True:
                i = 10
        number = 0
        numBytes = 1
        b = ba[offset]
        number = b & 127
        while b & 128 == 128:
            number = number << 7
            if number > self.maxValue:
                raise SDNVValueError(self.maxValue)
            b = ba[offset + numBytes]
            number += b & 127
            numBytes += 1
        if number > self.maxValue:
            raise SDNVValueError(self.maxValue)
        return (number, numBytes)
SDNVUtil = SDNV()

class SDNV2(Field):
    """ SDNV2 field """

    def addfield(self, pkt, s, val):
        if False:
            i = 10
            return i + 15
        return s + raw(SDNVUtil.encode(val))

    def getfield(self, pkt, s):
        if False:
            i = 10
            return i + 15
        b = bytearray(s)
        (val, len) = SDNVUtil.decode(b, 0)
        return (s[len:], val)

class SDNV2FieldLenField(FieldLenField, SDNV2):

    def addfield(self, pkt, s, val):
        if False:
            while True:
                i = 10
        return s + raw(SDNVUtil.encode(FieldLenField.i2m(self, pkt, val)))

class SDNV2LenField(LenField, SDNV2):

    def addfield(self, pkt, s, val):
        if False:
            return 10
        return s + raw(SDNVUtil.encode(LenField.i2m(self, pkt, val)))