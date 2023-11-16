SEED = 987654321

class Random(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.state0 = self.state1 = self.state2 = self.state3 = SEED

    def int32u(self):
        if False:
            for i in range(10):
                print('nop')
        self.state0 = (self.state0 & 4294967294) << 18 & 4294967295 ^ (self.state0 << 6 & 4294967295 ^ self.state0) >> 13
        self.state1 = (self.state1 & 4294967288) << 2 & 4294967295 ^ (self.state1 << 2 & 4294967295 ^ self.state1) >> 27
        self.state2 = (self.state2 & 4294967280) << 7 & 4294967295 ^ (self.state2 << 13 & 4294967295 ^ self.state2) >> 21
        self.state3 = (self.state3 & 4294967168) << 13 & 4294967295 ^ (self.state3 << 3 & 4294967295 ^ self.state3) >> 12
        return self.state0 ^ self.state1 ^ self.state2 ^ self.state3

    def real64(self):
        if False:
            print('Hello World!')
        (int0, int1) = (self.int32u(), self.int32u())
        return float(int0 < 2147483648 and int0 or int0 - 4294967296) * (1.0 / 4294967296.0) + 0.5 + float(int1 & 2097151) * (1.0 / 9007199254740992.0)