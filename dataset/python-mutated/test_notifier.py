import asyncio
from unittest.mock import MagicMock
import pytest
from tribler.core.utilities.notifier import Notifier, NotifierError

def test_add_remove_observer():
    if False:
        i = 10
        return i + 15
    notifier = Notifier()
    with pytest.raises(TypeError, match="^'topic' is not a callable object$"):
        notifier.add_observer('topic', lambda x: x)

    def topic1(x: int):
        if False:
            return 10
        pass
    with pytest.raises(TypeError, match="^'observer' is not a callable object$"):
        notifier.add_observer(topic1, 'observer')

    def observer1():
        if False:
            return 10
        pass
    with pytest.raises(TypeError, match='^Cannot add observer <function .*> to topic "topic1": the callback signature \\(\\) does not match the topic signature \\(x: int\\)$'):
        notifier.add_observer(topic1, observer1)

    def observer2(x):
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(TypeError, match='^Cannot add observer <function .*> to topic "topic1": the callback signature \\(x\\) does not match the topic signature \\(x: int\\)$'):
        notifier.add_observer(topic1, observer2)

    def observer3(x: str):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(TypeError, match='^Cannot add observer <function .*> to topic "topic1": the callback signature \\(x: str\\) does not match the topic signature \\(x: int\\)$'):
        notifier.add_observer(topic1, observer3)

    def observer4(y: int):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(TypeError, match='^Cannot add observer <function .*> to topic "topic1": the callback signature \\(y: int\\) does not match the topic signature \\(x: int\\)$'):
        notifier.add_observer(topic1, observer4)

    def observer5(x: int, y: int):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(TypeError, match='^Cannot add observer <function .*> to topic "topic1": the callback signature \\(x: int, y: int\\) does not match the topic signature \\(x: int\\)$'):
        notifier.add_observer(topic1, observer5)

    def observer6(x: int=None):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(TypeError, match='^Cannot add observer <function .*> to topic "topic1": the callback signature \\(x: int = None\\) does not match the topic signature \\(x: int\\)$'):
        notifier.add_observer(topic1, observer6)

    async def async1(x: int):
        pass
    with pytest.raises(TypeError, match='^Topic cannot be a coroutine function. Got: <function .*>$'):
        notifier.add_observer(async1, topic1)
    with pytest.raises(TypeError, match='^Observer cannot be a coroutine function. Got: <function .*>$'):
        notifier.add_observer(topic1, async1)
    with pytest.raises(TypeError, match='^Topic and observer cannot be the same function. Got: <function .*>$'):
        notifier.add_observer(topic1, topic1)

    def observer7(x: int):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(TypeError, match='^`synchronous` option may be True, False or None. Got: 1$'):
        notifier.add_observer(topic1, observer7, synchronous=1)
    with pytest.raises(TypeError, match='^synchronous=False option cannot be specified for a notifier without an event loop$'):
        notifier.add_observer(topic1, observer7, synchronous=False)
    assert not notifier.topics_by_name
    assert not notifier.topics
    assert not notifier.generic_observers
    assert not notifier.interceptors
    notifier.add_observer(topic1, observer7)
    assert notifier.topics_by_name == {'topic1': topic1}
    assert notifier.topics == {topic1: {observer7: True}}
    notifier.add_observer(topic1, observer7)
    assert notifier.topics == {topic1: {observer7: True}}

    def observer8(x: int):
        if False:
            print('Hello World!')
        pass
    notifier.add_observer(topic1, observer8)
    assert notifier.topics == {topic1: {observer7: True, observer8: True}}
    assert not notifier.generic_observers
    assert not notifier.interceptors

    def topic2(x: int):
        if False:
            for i in range(10):
                print('nop')
        pass

    def observer9(x: int):
        if False:
            print('Hello World!')
        pass
    notifier.remove_observer(topic2, observer7)
    notifier.remove_observer(topic1, observer9)
    assert notifier.topics == {topic1: {observer7: True, observer8: True}, topic2: {}}
    notifier.remove_observer(topic1, observer7)
    assert notifier.topics == {topic1: {observer8: True}, topic2: {}}
    notifier.remove_observer(topic1, observer8)
    assert notifier.topics == {topic1: {}, topic2: {}}

def test_two_topics_with_the_same_name():
    if False:
        return 10
    notifier = Notifier()

    def topic1(x: int):
        if False:
            i = 10
            return i + 15
        pass

    def observer1(x: int):
        if False:
            i = 10
            return i + 15
        pass
    notifier.add_observer(topic1, observer1)

    def topic1(x: int):
        if False:
            i = 10
            return i + 15
        pass

    def observer2(x: int):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(NotifierError, match='^Cannot register topic <.*topic1.*> because topic name topic1 is already taken by another topic <.*topic1.*>$'):
        notifier.add_observer(topic1, observer2)

def test_notify():
    if False:
        while True:
            i = 10

    def topic_a(a: int, b: str):
        if False:
            i = 10
            return i + 15
        pass

    def topic_b(x: int):
        if False:
            i = 10
            return i + 15
        pass
    calls = []

    def observer_a1(a: int, b: str):
        if False:
            return 10
        calls.append(('a1', a, b))

    def observer_a2(a: int, b: str):
        if False:
            return 10
        calls.append(('a2', a, b))

    def observer_b1(x: int):
        if False:
            print('Hello World!')
        calls.append(('b1', x))

    def generic_1(*args, **kwargs):
        if False:
            while True:
                i = 10
        calls.append(('generic1',) + args + (repr(kwargs),))

    def generic_2(*args, **kwargs):
        if False:
            print('Hello World!')
        calls.append(('generic2',) + args + (repr(kwargs),))
    notifier = Notifier()
    notifier.add_observer(topic_a, observer_a1)
    notifier.add_observer(topic_a, observer_a1)
    notifier.add_generic_observer(generic_1)
    with pytest.raises(TypeError):
        notifier[topic_a](123)
    assert calls == []
    notifier[topic_a](1, 'aaa')
    assert calls == [('generic1', topic_a, 1, 'aaa', '{}'), ('a1', 1, 'aaa')]
    calls.clear()
    notifier.add_observer(topic_a, observer_a2)
    notifier.add_observer(topic_b, observer_b1)
    notifier.add_generic_observer(generic_2)
    notifier[topic_a](2, 'bbb')
    assert calls == [('generic1', topic_a, 2, 'bbb', '{}'), ('generic2', topic_a, 2, 'bbb', '{}'), ('a1', 2, 'bbb'), ('a2', 2, 'bbb')]
    calls.clear()
    notifier[topic_b](x=111)
    assert calls == [('generic1', topic_b, "{'x': 111}"), ('generic2', topic_b, "{'x': 111}"), ('b1', 111)]
    calls.clear()
    notifier.logger.warning = MagicMock()
    notifier.notify_by_topic_name('non_existent_topic', x=1, y=2)
    notifier.logger.warning.assert_called_once_with('Topic with name `non_existent_topic` not found')
    notifier.notify_by_topic_name('topic_b', x=111)
    assert calls == [('generic1', topic_b, "{'x': 111}"), ('generic2', topic_b, "{'x': 111}"), ('b1', 111)]
    calls.clear()
    notifier.remove_observer(topic_b, observer_b1)
    notifier.remove_generic_observer(generic_1)
    notifier[topic_b](222)
    assert calls == [('generic2', topic_b, 222, '{}')]

async def test_notify_async(event_loop):

    def topic_a(a: int, b: str):
        if False:
            print('Hello World!')
        pass

    def topic_b(x: int):
        if False:
            i = 10
            return i + 15
        pass
    calls = []

    def observer_a1(a: int, b: str):
        if False:
            i = 10
            return i + 15
        calls.append(('a1', a, b))

    def observer_a2(a: int, b: str):
        if False:
            while True:
                i = 10
        calls.append(('a2', a, b))

    def observer_b1(x: int):
        if False:
            return 10
        calls.append(('b1', x))

    def generic_1(*args, **kwargs):
        if False:
            print('Hello World!')
        calls.append(('generic1',) + args + (repr(kwargs),))

    def generic_2(*args, **kwargs):
        if False:
            return 10
        calls.append(('generic2',) + args + (repr(kwargs),))
    notifier = Notifier(loop=event_loop)
    notifier.add_observer(topic_a, observer_a1)
    notifier.add_observer(topic_a, observer_a1)
    notifier.add_generic_observer(generic_1)
    with pytest.raises(NotifierError, match='^Cannot register the same observer with a different value of `synchronous` option$'):
        notifier.add_observer(topic_a, observer_a1, synchronous=True)
    with pytest.raises(TypeError):
        notifier[topic_a](123)
    assert calls == []
    notifier[topic_a](1, 'aaa')
    await asyncio.sleep(0.1)
    assert set(calls) == {('generic1', topic_a, 1, 'aaa', '{}'), ('a1', 1, 'aaa')}
    calls.clear()
    notifier.add_observer(topic_a, observer_a2)
    notifier.add_observer(topic_b, observer_b1)
    notifier.add_generic_observer(generic_2)
    notifier[topic_a](2, 'bbb')
    await asyncio.sleep(0.1)
    assert set(calls) == {('generic1', topic_a, 2, 'bbb', '{}'), ('generic2', topic_a, 2, 'bbb', '{}'), ('a1', 2, 'bbb'), ('a2', 2, 'bbb')}
    calls.clear()
    notifier[topic_b](x=111)
    await asyncio.sleep(0.1)
    assert set(calls) == {('generic1', topic_b, "{'x': 111}"), ('generic2', topic_b, "{'x': 111}"), ('b1', 111)}
    calls.clear()
    notifier.remove_observer(topic_b, observer_b1)
    notifier.remove_generic_observer(generic_1)
    notifier[topic_b](222)
    await asyncio.sleep(0.1)
    assert set(calls) == {('generic2', topic_b, 222, '{}')}

async def test_notify_with_exception(event_loop):

    def topic(x: int):
        if False:
            return 10
        pass
    calls = []

    def observer1(x: int):
        if False:
            return 10
        calls.append(('observer1', x))

    def observer2(x: int):
        if False:
            print('Hello World!')
        calls.append(('observer2', x))
        raise ZeroDivisionError

    def observer3(x: int):
        if False:
            i = 10
            return i + 15
        calls.append(('observer3', x))
    notifier = Notifier()
    notifier.add_observer(topic, observer1)
    notifier.add_observer(topic, observer2)
    notifier.add_observer(topic, observer3)
    notifier[topic](123)
    assert calls == [('observer1', 123), ('observer2', 123), ('observer3', 123)]
    calls.clear()
    notifier = Notifier(loop=event_loop)
    notifier.add_observer(topic, observer1)
    notifier.add_observer(topic, observer2)
    notifier.add_observer(topic, observer3)
    notifier[topic](123)
    assert calls == []
    await asyncio.sleep(0.1)
    assert set(calls) == {('observer1', 123), ('observer2', 123), ('observer3', 123)}

def test_notify_call_soon_threadsafe_with_exception(event_loop):
    if False:
        return 10
    notifier = Notifier(loop=event_loop)
    notifier.logger = MagicMock()
    notifier.loop = MagicMock(call_soon_threadsafe=MagicMock(side_effect=RuntimeError))

    def topic1(x: int):
        if False:
            while True:
                i = 10
        pass

    def observer1(x: int):
        if False:
            return 10
        pass
    notifier.add_observer(topic1, observer1)
    notifier[topic1](123)
    notifier.logger.warning.assert_called_once()