import json

class MessageTypes(object):
    (INVALID, MUST_SEND, BEST_EFFORT, SHUTDOWN) = range(1, 5)

class Message(object):

    def __init__(self, msg_type, payload):
        if False:
            while True:
                i = 10
        self.msg_type = msg_type
        self.payload = payload

    def serialize(self):
        if False:
            print('Hello World!')
        msg = {'msg_type': self.msg_type, 'payload': self.payload}
        return json.dumps(msg) + '\n'

    @staticmethod
    def deserialize(json_msg):
        if False:
            for i in range(10):
                print('nop')
        try:
            return Message(**json.loads(json_msg))
        except json.decoder.JSONDecodeError:
            return Message(MessageTypes.INVALID, None)