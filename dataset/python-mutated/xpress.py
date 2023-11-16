"""
@author:       Brendan Dolan-Gavitt
@license:      GNU General Public License 2.0
@contact:      bdolangavitt@wesleyan.edu
"""
from struct import unpack
from struct import error as StructError

def recombine(outbuf):
    if False:
        return 10
    return ''.join((outbuf[k] for k in sorted(outbuf.keys())))

def xpress_decode(inputBuffer):
    if False:
        i = 10
        return i + 15
    outputBuffer = {}
    outputIndex = 0
    inputIndex = 0
    indicatorBit = 0
    nibbleIndex = 0
    while inputIndex < len(inputBuffer):
        if indicatorBit == 0:
            try:
                indicator = unpack('<L', inputBuffer[inputIndex:inputIndex + 4])[0]
            except StructError:
                return recombine(outputBuffer)
            inputIndex += 4
            indicatorBit = 32
        indicatorBit = indicatorBit - 1
        if not indicator & 1 << indicatorBit:
            try:
                outputBuffer[outputIndex] = inputBuffer[inputIndex]
            except IndexError:
                return recombine(outputBuffer)
            inputIndex += 1
            outputIndex += 1
        else:
            try:
                length = unpack('<H', inputBuffer[inputIndex:inputIndex + 2])[0]
            except StructError:
                return recombine(outputBuffer)
            inputIndex += 2
            offset = length / 8
            length = length % 8
            if length == 7:
                if nibbleIndex == 0:
                    nibbleIndex = inputIndex
                    length = ord(inputBuffer[inputIndex]) % 16
                    inputIndex += 1
                else:
                    length = ord(inputBuffer[nibbleIndex]) / 16
                    nibbleIndex = 0
                if length == 15:
                    length = ord(inputBuffer[inputIndex])
                    inputIndex += 1
                    if length == 255:
                        try:
                            length = unpack('<H', inputBuffer[inputIndex:inputIndex + 2])[0]
                        except StructError:
                            return recombine(outputBuffer)
                        inputIndex = inputIndex + 2
                        length = length - (15 + 7)
                    length = length + 15
                length = length + 7
            length = length + 3
            while length != 0:
                try:
                    outputBuffer[outputIndex] = outputBuffer[outputIndex - offset - 1]
                except KeyError:
                    return recombine(outputBuffer)
                outputIndex += 1
                length -= 1
    return recombine(outputBuffer)
try:
    import pyxpress
    xpress_decode = pyxpress.decode
except ImportError:
    pass
if __name__ == '__main__':
    import sys
    dec_data = xpress_decode(open(sys.argv[1]).read())
    sys.stdout.write(dec_data)