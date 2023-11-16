from hscommon.testutil import eq_
from hscommon.notify import Broadcaster, Listener, Repeater

class HelloListener(Listener):

    def __init__(self, broadcaster):
        if False:
            return 10
        Listener.__init__(self, broadcaster)
        self.hello_count = 0

    def hello(self):
        if False:
            for i in range(10):
                print('nop')
        self.hello_count += 1

class HelloRepeater(Repeater):

    def __init__(self, broadcaster):
        if False:
            while True:
                i = 10
        Repeater.__init__(self, broadcaster)
        self.hello_count = 0

    def hello(self):
        if False:
            while True:
                i = 10
        self.hello_count += 1

def create_pair():
    if False:
        return 10
    b = Broadcaster()
    listener = HelloListener(b)
    return (b, listener)

def test_disconnect_during_notification():
    if False:
        i = 10
        return i + 15

    class Disconnecter(Listener):

        def __init__(self, broadcaster):
            if False:
                i = 10
                return i + 15
            Listener.__init__(self, broadcaster)
            self.hello_count = 0

        def hello(self):
            if False:
                for i in range(10):
                    print('nop')
            self.hello_count += 1
            self.other.disconnect()
    broadcaster = Broadcaster()
    first = Disconnecter(broadcaster)
    second = Disconnecter(broadcaster)
    (first.other, second.other) = (second, first)
    first.connect()
    second.connect()
    broadcaster.notify('hello')
    eq_(first.hello_count + second.hello_count, 1)

def test_disconnect():
    if False:
        return 10
    (b, listener) = create_pair()
    listener.connect()
    listener.disconnect()
    b.notify('hello')
    eq_(listener.hello_count, 0)

def test_disconnect_when_not_connected():
    if False:
        while True:
            i = 10
    (b, listener) = create_pair()
    listener.disconnect()

def test_not_connected_on_init():
    if False:
        for i in range(10):
            print('nop')
    (b, listener) = create_pair()
    b.notify('hello')
    eq_(listener.hello_count, 0)

def test_notify():
    if False:
        while True:
            i = 10
    (b, listener) = create_pair()
    listener.connect()
    b.notify('hello')
    eq_(listener.hello_count, 1)

def test_reconnect():
    if False:
        i = 10
        return i + 15
    (b, listener) = create_pair()
    listener.connect()
    listener.disconnect()
    listener.connect()
    b.notify('hello')
    eq_(listener.hello_count, 1)

def test_repeater():
    if False:
        i = 10
        return i + 15
    b = Broadcaster()
    r = HelloRepeater(b)
    listener = HelloListener(r)
    r.connect()
    listener.connect()
    b.notify('hello')
    eq_(r.hello_count, 1)
    eq_(listener.hello_count, 1)

def test_repeater_with_repeated_notifications():
    if False:
        for i in range(10):
            print('nop')

    class MyRepeater(HelloRepeater):
        REPEATED_NOTIFICATIONS = {'hello'}

        def __init__(self, broadcaster):
            if False:
                while True:
                    i = 10
            HelloRepeater.__init__(self, broadcaster)
            self.foo_count = 0

        def foo(self):
            if False:
                for i in range(10):
                    print('nop')
            self.foo_count += 1
    b = Broadcaster()
    r = MyRepeater(b)
    listener = HelloListener(r)
    r.connect()
    listener.connect()
    b.notify('hello')
    b.notify('foo')
    eq_(r.hello_count, 1)
    eq_(listener.hello_count, 1)
    eq_(r.foo_count, 1)

def test_repeater_doesnt_try_to_dispatch_to_self_if_it_cant():
    if False:
        print('Hello World!')
    b = Broadcaster()
    r = Repeater(b)
    listener = HelloListener(r)
    r.connect()
    listener.connect()
    b.notify('hello')
    eq_(listener.hello_count, 1)

def test_bind_messages():
    if False:
        return 10
    (b, listener) = create_pair()
    listener.bind_messages({'foo', 'bar'}, listener.hello)
    listener.connect()
    b.notify('foo')
    b.notify('bar')
    b.notify('hello')
    eq_(listener.hello_count, 3)