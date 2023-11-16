class RC4:

    def __init__(self, key):
        if False:
            while True:
                i = 10
        bkey = bytearray(key)
        j = 0
        self.state = bytearray(range(256))
        for i in range(256):
            j = j + self.state[i] + bkey[i % len(key)] & 255
            (self.state[i], self.state[j]) = (self.state[j], self.state[i])

    def encrypt(self, data):
        if False:
            return 10
        i = j = 0
        out = bytearray()
        for char in bytearray(data):
            i = i + 1 & 255
            j = j + self.state[i] & 255
            (self.state[i], self.state[j]) = (self.state[j], self.state[i])
            out.append(char ^ self.state[self.state[i] + self.state[j] & 255])
        return bytes(out)

    def decrypt(self, data):
        if False:
            while True:
                i = 10
        return self.encrypt(data)