"""
STK500v2 protocol implementation for programming AVR chips.
The STK500v2 protocol is used by the ArduinoMega2560 and a few other Arduino platforms to load firmware.
This is a python 3 conversion of the code created by David Braam for the Cura project.
"""
import struct
import sys
import time
from serial import Serial
from serial import SerialException
from serial import SerialTimeoutException
from UM.Logger import Logger
from . import ispBase, intelHex

class Stk500v2(ispBase.IspBase):

    def __init__(self):
        if False:
            return 10
        self.serial = None
        self.seq = 1
        self.last_addr = -1
        self.progress_callback = None

    def connect(self, port='COM22', speed=115200):
        if False:
            i = 10
            return i + 15
        if self.serial is not None:
            self.close()
        try:
            self.serial = Serial(str(port), speed, timeout=1, writeTimeout=10000)
        except SerialException:
            raise ispBase.IspError('Failed to open serial port')
        except:
            raise ispBase.IspError('Unexpected error while connecting to serial port:' + port + ':' + str(sys.exc_info()[0]))
        self.seq = 1
        for n in range(0, 2):
            self.serial.setDTR(True)
            time.sleep(0.1)
            self.serial.setDTR(False)
            time.sleep(0.1)
        time.sleep(0.2)
        self.serial.flushInput()
        self.serial.flushOutput()
        try:
            if self.sendMessage([16, 200, 100, 25, 32, 0, 83, 3, 172, 83, 0, 0]) != [16, 0]:
                raise ispBase.IspError('Failed to enter programming mode')
            self.sendMessage([6, 128, 0, 0, 0])
            if self.sendMessage([238])[1] == 0:
                self._has_checksum = True
            else:
                self._has_checksum = False
        except ispBase.IspError:
            self.close()
            raise
        self.serial.timeout = 5

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def leaveISP(self):
        if False:
            for i in range(10):
                print('nop')
        if self.serial is not None:
            if self.sendMessage([17]) != [17, 0]:
                raise ispBase.IspError('Failed to leave programming mode')
            ret = self.serial
            self.serial = None
            return ret
        return None

    def isConnected(self):
        if False:
            return 10
        return self.serial is not None

    def hasChecksumFunction(self):
        if False:
            for i in range(10):
                print('nop')
        return self._has_checksum

    def sendISP(self, data):
        if False:
            print('Hello World!')
        recv = self.sendMessage([29, 4, 4, 0, data[0], data[1], data[2], data[3]])
        return recv[2:6]

    def writeFlash(self, flash_data):
        if False:
            for i in range(10):
                print('nop')
        page_size = self.chip['pageSize'] * 2
        flash_size = page_size * self.chip['pageCount']
        Logger.log('d', 'Writing flash')
        if flash_size > 65535:
            self.sendMessage([6, 128, 0, 0, 0])
        else:
            self.sendMessage([6, 0, 0, 0, 0])
        load_count = (len(flash_data) + page_size - 1) / page_size
        for i in range(0, int(load_count)):
            self.sendMessage([19, page_size >> 8, page_size & 255, 193, 10, 64, 76, 32, 0, 0] + flash_data[i * page_size:i * page_size + page_size])
            if self.progress_callback is not None:
                if self._has_checksum:
                    self.progress_callback(i + 1, load_count)
                else:
                    self.progress_callback(i + 1, load_count * 2)

    def verifyFlash(self, flash_data):
        if False:
            for i in range(10):
                print('nop')
        if self._has_checksum:
            self.sendMessage([6, 0, len(flash_data) >> 17 & 255, len(flash_data) >> 9 & 255, len(flash_data) >> 1 & 255])
            res = self.sendMessage([238])
            checksum_recv = res[2] | res[3] << 8
            checksum = 0
            for d in flash_data:
                checksum += d
            checksum &= 65535
            if hex(checksum) != hex(checksum_recv):
                raise ispBase.IspError('Verify checksum mismatch: 0x%x != 0x%x' % (checksum & 65535, checksum_recv))
        else:
            flash_size = self.chip['pageSize'] * 2 * self.chip['pageCount']
            if flash_size > 65535:
                self.sendMessage([6, 128, 0, 0, 0])
            else:
                self.sendMessage([6, 0, 0, 0, 0])
            load_count = (len(flash_data) + 255) / 256
            for i in range(0, int(load_count)):
                recv = self.sendMessage([20, 1, 0, 32])[2:258]
                if self.progress_callback is not None:
                    self.progress_callback(load_count + i + 1, load_count * 2)
                for j in range(0, 256):
                    if i * 256 + j < len(flash_data) and flash_data[i * 256 + j] != recv[j]:
                        raise ispBase.IspError('Verify error at: 0x%x' % (i * 256 + j))

    def sendMessage(self, data):
        if False:
            for i in range(10):
                print('nop')
        message = struct.pack('>BBHB', 27, self.seq, len(data), 14)
        for c in data:
            message += struct.pack('>B', c)
        checksum = 0
        for c in message:
            checksum ^= c
        message += struct.pack('>B', checksum)
        try:
            self.serial.write(message)
            self.serial.flush()
        except SerialTimeoutException:
            raise ispBase.IspError('Serial send timeout')
        self.seq = self.seq + 1 & 255
        return self.recvMessage()

    def recvMessage(self):
        if False:
            return 10
        state = 'Start'
        checksum = 0
        while True:
            s = self.serial.read()
            if len(s) < 1:
                raise ispBase.IspError('Timeout')
            b = struct.unpack('>B', s)[0]
            checksum ^= b
            if state == 'Start':
                if b == 27:
                    state = 'GetSeq'
                    checksum = 27
            elif state == 'GetSeq':
                state = 'MsgSize1'
            elif state == 'MsgSize1':
                msg_size = b << 8
                state = 'MsgSize2'
            elif state == 'MsgSize2':
                msg_size |= b
                state = 'Token'
            elif state == 'Token':
                if b != 14:
                    state = 'Start'
                else:
                    state = 'Data'
                    data = []
            elif state == 'Data':
                data.append(b)
                if len(data) == msg_size:
                    state = 'Checksum'
            elif state == 'Checksum':
                if checksum != 0:
                    state = 'Start'
                else:
                    return data

def portList():
    if False:
        print('Hello World!')
    ret = []
    import _winreg
    key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, 'HARDWARE\\DEVICEMAP\\SERIALCOMM')
    i = 0
    while True:
        try:
            values = _winreg.EnumValue(key, i)
        except:
            return ret
        if 'USBSER' in values[0]:
            ret.append(values[1])
        i += 1
    return ret

def runProgrammer(port, filename):
    if False:
        i = 10
        return i + 15
    " Run an STK500v2 program on serial port 'port' and write 'filename' into flash. "
    programmer = Stk500v2()
    programmer.connect(port=port)
    programmer.programChip(intelHex.readHex(filename))
    programmer.close()

def main():
    if False:
        for i in range(10):
            print('nop')
    ' Entry point to call the stk500v2 programmer from the commandline. '
    import threading
    if sys.argv[1] == 'AUTO':
        Logger.log('d', 'portList(): ', repr(portList()))
        for port in portList():
            threading.Thread(target=runProgrammer, args=(port, sys.argv[2])).start()
            time.sleep(5)
    else:
        programmer = Stk500v2()
        programmer.connect(port=sys.argv[1])
        programmer.programChip(intelHex.readHex(sys.argv[2]))
        sys.exit(1)
if __name__ == '__main__':
    main()