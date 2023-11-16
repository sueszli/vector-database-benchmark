class FrameHeader:
    fin: int
    rsv1: int
    rsv2: int
    rsv3: int
    opcode: int
    masked: int
    length: int
    OPCODE_CONTINUATION = 0
    OPCODE_TEXT = 1
    OPCODE_BINARY = 2
    OPCODE_CLOSE = 8
    OPCODE_PING = 9
    OPCODE_PONG = 10

    def __init__(self, opcode: int, fin: int=1, rsv1: int=0, rsv2: int=0, rsv3: int=0, masked: int=0, length: int=0):
        if False:
            for i in range(10):
                print('nop')
        self.opcode = opcode
        self.fin = fin
        self.rsv1 = rsv1
        self.rsv2 = rsv2
        self.rsv3 = rsv3
        self.masked = masked
        self.length = length