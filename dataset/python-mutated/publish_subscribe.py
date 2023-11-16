"""
Reference:
http://www.slideshare.net/ishraqabd/publish-subscribe-model-overview-13368808
Author: https://github.com/HanWenfang
"""
from __future__ import annotations

class Provider:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.msg_queue = []
        self.subscribers = {}

    def notify(self, msg: str) -> None:
        if False:
            while True:
                i = 10
        self.msg_queue.append(msg)

    def subscribe(self, msg: str, subscriber: Subscriber) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.subscribers.setdefault(msg, []).append(subscriber)

    def unsubscribe(self, msg: str, subscriber: Subscriber) -> None:
        if False:
            i = 10
            return i + 15
        self.subscribers[msg].remove(subscriber)

    def update(self) -> None:
        if False:
            i = 10
            return i + 15
        for msg in self.msg_queue:
            for sub in self.subscribers.get(msg, []):
                sub.run(msg)
        self.msg_queue = []

class Publisher:

    def __init__(self, msg_center: Provider) -> None:
        if False:
            i = 10
            return i + 15
        self.provider = msg_center

    def publish(self, msg: str) -> None:
        if False:
            return 10
        self.provider.notify(msg)

class Subscriber:

    def __init__(self, name: str, msg_center: Provider) -> None:
        if False:
            print('Hello World!')
        self.name = name
        self.provider = msg_center

    def subscribe(self, msg: str) -> None:
        if False:
            while True:
                i = 10
        self.provider.subscribe(msg, self)

    def unsubscribe(self, msg: str) -> None:
        if False:
            return 10
        self.provider.unsubscribe(msg, self)

    def run(self, msg: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        print(f'{self.name} got {msg}')

def main():
    if False:
        while True:
            i = 10
    '\n    >>> message_center = Provider()\n\n    >>> fftv = Publisher(message_center)\n\n    >>> jim = Subscriber("jim", message_center)\n    >>> jim.subscribe("cartoon")\n    >>> jack = Subscriber("jack", message_center)\n    >>> jack.subscribe("music")\n    >>> gee = Subscriber("gee", message_center)\n    >>> gee.subscribe("movie")\n    >>> vani = Subscriber("vani", message_center)\n    >>> vani.subscribe("movie")\n    >>> vani.unsubscribe("movie")\n\n    # Note that no one subscribed to `ads`\n    # and that vani changed their mind\n\n    >>> fftv.publish("cartoon")\n    >>> fftv.publish("music")\n    >>> fftv.publish("ads")\n    >>> fftv.publish("movie")\n    >>> fftv.publish("cartoon")\n    >>> fftv.publish("cartoon")\n    >>> fftv.publish("movie")\n    >>> fftv.publish("blank")\n\n    >>> message_center.update()\n    jim got cartoon\n    jack got music\n    gee got movie\n    jim got cartoon\n    jim got cartoon\n    gee got movie\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()