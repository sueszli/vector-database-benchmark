class JadeError(Exception):
    INVALID_REQUEST = -32600
    UNKNOWN_METHOD = -32601
    BAD_PARAMETERS = -32602
    INTERNAL_ERROR = -32603
    USER_CANCELLED = -32000
    PROTOCOL_ERROR = -32001
    HW_LOCKED = -32002
    NETWORK_MISMATCH = -32003

    def __init__(self, code, message, data):
        if False:
            print('Hello World!')
        self.code = code
        self.message = message
        self.data = data

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'JadeError: ' + str(self.code) + ' - ' + self.message + ' (Data: ' + repr(self.data) + ')'

    def __str__(self):
        if False:
            while True:
                i = 10
        return repr(self)