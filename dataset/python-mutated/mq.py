from typing import Tuple

class MQ:
    """
    Overview:
        Abstract basic mq class.
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            The __init__ method of the inheritance must support the extra kwargs parameter.\n        '
        pass

    def listen(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Bind to local socket or connect to third party components.\n        '
        raise NotImplementedError

    def publish(self, topic: str, data: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Send data to mq.\n        Arguments:\n            - topic (:obj:`str`): Topic.\n            - data (:obj:`bytes`): Payload data.\n        '
        raise NotImplementedError

    def subscribe(self, topic: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Subscribe to the topic.\n        Arguments:\n            - topic (:obj:`str`): Topic\n        '
        raise NotImplementedError

    def unsubscribe(self, topic: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Unsubscribe from the topic.\n        Arguments:\n            - topic (:obj:`str`): Topic\n        '
        raise NotImplementedError

    def recv(self) -> Tuple[str, bytes]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Wait for incoming message, this function will block the current thread.\n        Returns:\n            - data (:obj:`Any`): The sent payload.\n        '
        raise NotImplementedError

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Unsubscribe from all topics and stop the connection to the message queue server.\n        '
        return