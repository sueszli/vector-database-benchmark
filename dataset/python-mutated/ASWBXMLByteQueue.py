"""
@author: David Shaw, shawd@vmware.com

Inspired by EAS Inspector for Fiddler
https://easinspectorforfiddler.codeplex.com

----- The MIT License (MIT) -----
Filename: ASWBXMLByteQueue.py
Copyright (c) 2014, David P. Shaw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from queue import Queue
import logging

class ASWBXMLByteQueue(Queue):

    def __init__(self, wbxmlBytes):
        if False:
            for i in range(10):
                print('nop')
        self.bytesDequeued = 0
        self.bytesEnqueued = 0
        Queue.__init__(self)
        for byte in wbxmlBytes:
            self.put(byte)
            self.bytesEnqueued += 1
        logging.debug('Array byte count: %d, enqueued: %d' % (self.qsize(), self.bytesEnqueued))
    '\n    Created to debug the dequeueing of bytes\n    '

    def dequeueAndLog(self):
        if False:
            while True:
                i = 10
        singleByte = self.get()
        self.bytesDequeued += 1
        logging.debug('Dequeued byte 0x{0:X} ({1} total)'.format(singleByte, self.bytesDequeued))
        return singleByte
    '\n    Return true if the continuation bit is set in the byte\n    '

    def checkContinuationBit(self, byteval):
        if False:
            return 10
        continuationBitmask = 128
        return continuationBitmask & byteval != 0

    def dequeueMultibyteInt(self):
        if False:
            i = 10
            return i + 15
        iReturn = 0
        singleByte = 255
        while True:
            iReturn <<= 7
            if self.qsize() == 0:
                break
            else:
                singleByte = self.dequeueAndLog()
            iReturn += int(singleByte & 127)
            if not self.checkContinuationBit(singleByte):
                return iReturn

    def dequeueString(self, length=None):
        if False:
            return 10
        if length != None:
            currentByte = 0
            strReturn = ''
            for i in range(0, length):
                if self.qsize() == 0:
                    break
                currentByte = self.dequeueAndLog()
                strReturn += chr(currentByte)
        else:
            currentByte = 0
            strReturn = ''
            while True:
                currentByte = self.dequeueAndLog()
                if currentByte != 0:
                    strReturn += chr(currentByte)
                else:
                    break
        return strReturn