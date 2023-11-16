import sys
import binascii
import hashlib

class ApiRosClient(object):
    """RouterOS API"""

    def __init__(self, sk):
        if False:
            while True:
                i = 10
        self.sk = sk
        self.currenttag = 0

    def login(self, username, pwd):
        if False:
            return 10
        for (repl, attrs) in self.talk(['/login']):
            chal = binascii.unhexlify(attrs['=ret'].encode('UTF-8'))
        md = hashlib.md5()
        md.update(b'\x00')
        md.update(pwd.encode('UTF-8'))
        md.update(chal)
        output = self.talk(['/login', '=name=' + username, '=response=00' + binascii.hexlify(md.digest()).decode('UTF-8')])
        return output

    def talk(self, words):
        if False:
            while True:
                i = 10
        if self.writeSentence(words) == 0:
            return
        r = []
        while 1:
            i = self.readSentence()
            if len(i) == 0:
                continue
            reply = i[0]
            attrs = {}
            for w in i[1:]:
                j = w.find('=', 1)
                if j == -1:
                    attrs[w] = ''
                else:
                    attrs[w[:j]] = w[j + 1:]
            r.append((reply, attrs))
            if reply == '!done':
                return r

    def writeSentence(self, words):
        if False:
            print('Hello World!')
        ret = 0
        for w in words:
            self.writeWord(w)
            ret += 1
        self.writeWord('')
        return ret

    def readSentence(self):
        if False:
            while True:
                i = 10
        r = []
        while 1:
            w = self.readWord()
            if w == '':
                return r
            r.append(w)

    def writeWord(self, w):
        if False:
            while True:
                i = 10
        self.writeLen(len(w))
        self.writeStr(w)

    def readWord(self):
        if False:
            for i in range(10):
                print('nop')
        ret = self.readStr(self.readLen())
        return ret

    def writeLen(self, length):
        if False:
            i = 10
            return i + 15
        if length < 128:
            self.writeByte(length.to_bytes(1, sys.byteorder))
        elif length < 16384:
            length |= 32768
            self.writeByte((length >> 8 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length & 255).to_bytes(1, sys.byteorder))
        elif length < 2097152:
            length |= 12582912
            self.writeByte((length >> 16 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length >> 8 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length & 255).to_bytes(1, sys.byteorder))
        elif length < 268435456:
            length |= 3758096384
            self.writeByte((length >> 24 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length >> 16 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length >> 8 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length & 255).to_bytes(1, sys.byteorder))
        else:
            self.writeByte(240 .to_bytes(1, sys.byteorder))
            self.writeByte((length >> 24 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length >> 16 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length >> 8 & 255).to_bytes(1, sys.byteorder))
            self.writeByte((length & 255).to_bytes(1, sys.byteorder))

    def readLen(self):
        if False:
            i = 10
            return i + 15
        c = ord(self.readStr(1))
        if c & 128 == 0:
            pass
        elif c & 192 == 128:
            c &= ~192
            c <<= 8
            c += ord(self.readStr(1))
        elif c & 224 == 192:
            c &= ~224
            c <<= 8
            c += ord(self.readStr(1))
            c <<= 8
            c += ord(self.readStr(1))
        elif c & 240 == 224:
            c &= ~240
            c <<= 8
            c += ord(self.readStr(1))
            c <<= 8
            c += ord(self.readStr(1))
            c <<= 8
            c += ord(self.readStr(1))
        elif c & 248 == 240:
            c = ord(self.readStr(1))
            c <<= 8
            c += ord(self.readStr(1))
            c <<= 8
            c += ord(self.readStr(1))
            c <<= 8
            c += ord(self.readStr(1))
        return c

    def writeStr(self, str):
        if False:
            while True:
                i = 10
        n = 0
        while n < len(str):
            r = self.sk.send(bytes(str[n:], 'UTF-8'))
            if r == 0:
                raise RuntimeError('connection closed by remote end')
            n += r

    def writeByte(self, str):
        if False:
            while True:
                i = 10
        n = 0
        while n < len(str):
            r = self.sk.send(str[n:])
            if r == 0:
                raise RuntimeError('connection closed by remote end')
            n += r

    def readStr(self, length):
        if False:
            i = 10
            return i + 15
        ret = ''
        while len(ret) < length:
            s = self.sk.recv(length - len(ret))
            if s == '':
                raise RuntimeError('connection closed by remote end')
            ret += s.decode('UTF-8', 'replace')
        return ret