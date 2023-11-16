from panda3d.core import DatagramIterator
from panda3d.direct import STInt8, STInt16, STInt32, STInt64, STUint8, STUint16, STUint32, STUint64, STFloat64, STString, STBlob, STBlob32, STInt16array, STInt32array, STUint16array, STUint32array, STInt8array, STUint8array, STUint32uint8array

class PyDatagramIterator(DatagramIterator):
    FuncDict = {STInt8: DatagramIterator.getInt8, STInt16: DatagramIterator.getInt16, STInt32: DatagramIterator.getInt32, STInt64: DatagramIterator.getInt64, STUint8: DatagramIterator.getUint8, STUint16: DatagramIterator.getUint16, STUint32: DatagramIterator.getUint32, STUint64: DatagramIterator.getUint64, STFloat64: DatagramIterator.getFloat64, STString: DatagramIterator.getString, STBlob: DatagramIterator.getBlob, STBlob32: DatagramIterator.getBlob32}
    getChannel = DatagramIterator.getUint64

    def __init__(self, datagram=None, offset=0):
        if False:
            for i in range(10):
                print('nop')
        if datagram is not None:
            super().__init__(datagram, offset)
            self.__initialDatagram = datagram
        else:
            super().__init__()

    def assign(self, datagram, offset=0):
        if False:
            for i in range(10):
                print('nop')
        super().assign(datagram, offset)
        self.__initialDatagram = datagram

    def getArg(self, subatomicType, divisor=1):
        if False:
            for i in range(10):
                print('nop')
        if divisor == 1:
            getFunc = self.FuncDict.get(subatomicType)
            if getFunc:
                retVal = getFunc(self)
            elif subatomicType == STInt8array:
                len = self.getUint16()
                retVal = []
                for i in range(len):
                    retVal.append(self.getInt8())
            elif subatomicType == STInt16array:
                len = self.getUint16() >> 1
                retVal = []
                for i in range(len):
                    retVal.append(self.getInt16())
            elif subatomicType == STInt32array:
                len = self.getUint16() >> 2
                retVal = []
                for i in range(len):
                    retVal.append(self.getInt32())
            elif subatomicType == STUint8array:
                len = self.getUint16()
                retVal = []
                for i in range(len):
                    retVal.append(self.getUint8())
            elif subatomicType == STUint16array:
                len = self.getUint16() >> 1
                retVal = []
                for i in range(len):
                    retVal.append(self.getUint16())
            elif subatomicType == STUint32array:
                len = self.getUint16() >> 2
                retVal = []
                for i in range(len):
                    retVal.append(self.getUint32())
            elif subatomicType == STUint32uint8array:
                len = self.getUint16() / 5
                retVal = []
                for i in range(len):
                    a = self.getUint32()
                    b = self.getUint8()
                    retVal.append((a, b))
            else:
                raise Exception('Error: No such type as: ' + str(subatomicType))
        else:
            getFunc = self.FuncDict.get(subatomicType)
            if getFunc:
                retVal = getFunc(self) / float(divisor)
            elif subatomicType == STInt8array:
                len = self.getUint8() >> 1
                retVal = []
                for i in range(len):
                    retVal.append(self.getInt8() / float(divisor))
            elif subatomicType == STInt16array:
                len = self.getUint16() >> 1
                retVal = []
                for i in range(len):
                    retVal.append(self.getInt16() / float(divisor))
            elif subatomicType == STInt32array:
                len = self.getUint16() >> 2
                retVal = []
                for i in range(len):
                    retVal.append(self.getInt32() / float(divisor))
            elif subatomicType == STUint8array:
                len = self.getUint8() >> 1
                retVal = []
                for i in range(len):
                    retVal.append(self.getUint8() / float(divisor))
            elif subatomicType == STUint16array:
                len = self.getUint16() >> 1
                retVal = []
                for i in range(len):
                    retVal.append(self.getUint16() / float(divisor))
            elif subatomicType == STUint32array:
                len = self.getUint16() >> 2
                retVal = []
                for i in range(len):
                    retVal.append(self.getUint32() / float(divisor))
            elif subatomicType == STUint32uint8array:
                len = self.getUint16() / 5
                retVal = []
                for i in range(len):
                    a = self.getUint32()
                    b = self.getUint8()
                    retVal.append((a / float(divisor), b / float(divisor)))
            else:
                raise Exception('Error: No such type as: ' + str(subatomicType))
        return retVal