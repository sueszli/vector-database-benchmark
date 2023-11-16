import pytest
from random import choice
from string import ascii_uppercase as str_alpha
from string import digits as str_num
from apprise import NotifyBase
from apprise.common import NotifyFormat
from apprise.common import OverflowMode
import logging
logging.disable(logging.CRITICAL)

def test_notify_overflow_truncate():
    if False:
        while True:
            i = 10
    '\n    API: Overflow Truncate Functionality Testing\n\n    '
    row = 24
    body_len = 1024
    title_len = 1024
    body = ''.join((choice(str_alpha + str_num + ' ') for _ in range(body_len)))
    body = '\r\n'.join([body[i:i + row] for i in range(0, len(body), row)])
    body = body[0:1024]
    title = ''.join((choice(str_alpha + str_num) for _ in range(title_len)))

    class TestNotification(NotifyBase):
        title_maxlen = 10

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                return 10
            return True
    with pytest.raises(TypeError):
        obj = TestNotification(overflow='invalid')
    obj = TestNotification(overflow=OverflowMode.TRUNCATE)
    assert obj is not None
    chunks = obj._apply_overflow(body=body, title=title, overflow=None)
    chunks = obj._apply_overflow(body=body, title=title, overflow=OverflowMode.SPLIT)
    assert len(chunks) == 1
    assert body.rstrip() == chunks[0].get('body')
    assert title[0:TestNotification.title_maxlen] == chunks[0].get('title')

    class TestNotification(NotifyBase):
        title_maxlen = 5
        body_max_line_count = 5

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return True
    obj = TestNotification(overflow=OverflowMode.TRUNCATE)
    assert obj is not None
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    assert len(chunks[0].get('body').split('\n')) == TestNotification.body_max_line_count
    assert title[0:TestNotification.title_maxlen] == chunks[0].get('title')

    class TestNotification(NotifyBase):
        title_maxlen = title_len
        body_maxlen = 10

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                return 10
            return True
    obj = TestNotification(overflow=OverflowMode.TRUNCATE)
    assert obj is not None
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    assert body[0:TestNotification.body_maxlen] == chunks[0].get('body')
    assert title == chunks[0].get('title')

    class TestNotification(NotifyBase):
        title_maxlen = 0
        body_maxlen = 100

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                print('Hello World!')
            return True
    obj = TestNotification(overflow=OverflowMode.TRUNCATE)
    assert obj is not None
    obj.notify_format = NotifyFormat.HTML
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    obj.notify_format = NotifyFormat.MARKDOWN
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    obj.notify_format = NotifyFormat.TEXT
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    assert len(chunks[0].get('body')) == TestNotification.body_maxlen
    assert title[0:TestNotification.body_maxlen] == chunks[0].get('body')

def test_notify_overflow_split():
    if False:
        print('Hello World!')
    '\n    API: Overflow Split Functionality Testing\n\n    '
    row = 24
    body_len = 1024
    title_len = 1024
    body = ''.join((choice(str_alpha + str_num) for _ in range(body_len)))
    body = '\r\n'.join([body[i:i + row] for i in range(0, len(body), row)])
    body = body[0:1024]
    title = ''.join((choice(str_alpha + str_num) for _ in range(title_len)))

    class TestNotification(NotifyBase):
        title_maxlen = 10

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return True
    obj = TestNotification(overflow=OverflowMode.SPLIT)
    assert obj is not None
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    assert body == chunks[0].get('body')
    assert title[0:TestNotification.title_maxlen] == chunks[0].get('title')

    class TestNotification(NotifyBase):
        title_maxlen = 5
        body_max_line_count = 5

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return True
    obj = TestNotification(overflow=OverflowMode.SPLIT)
    assert obj is not None
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    assert len(chunks[0].get('body').split('\n')) == TestNotification.body_max_line_count
    assert title[0:TestNotification.title_maxlen] == chunks[0].get('title')

    class TestNotification(NotifyBase):
        title_maxlen = title_len
        body_maxlen = int(body_len / 4)

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                print('Hello World!')
            return True
    obj = TestNotification(overflow=OverflowMode.SPLIT)
    assert obj is not None
    chunks = obj._apply_overflow(body=body, title=title)
    offset = 0
    assert len(chunks) == 4
    for chunk in chunks:
        assert title == chunk.get('title')
        _body = chunk.get('body')
        assert body[offset:len(_body) + offset].rstrip() == _body
        offset += len(_body)

    class TestNotification(NotifyBase):
        title_maxlen = 0
        body_maxlen = int(title_len / 4)

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                print('Hello World!')
            return True
    obj = TestNotification(overflow=OverflowMode.SPLIT)
    assert obj is not None
    chunks = obj._apply_overflow(body=body, title=title)
    offset = 0
    bulk = title + '\r\n' + body
    assert len(chunks) == int(len(bulk) / TestNotification.body_maxlen) + (1 if len(bulk) % TestNotification.body_maxlen else 0)
    for chunk in chunks:
        assert chunk.get('title') == ''
        _body = chunk.get('body')
        assert bulk[offset:len(_body) + offset] == _body
        offset += len(_body)

def test_notify_overflow_general():
    if False:
        for i in range(10):
            print('nop')
    '\n    API: Overflow General Testing\n\n    '

    class TestMarkdownNotification(NotifyBase):
        title_maxlen = 0
        notify_format = NotifyFormat.MARKDOWN

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            super().__init__(**kwargs)

        def notify(self, *args, **kwargs):
            if False:
                return 10
            return True
    obj = TestMarkdownNotification()
    assert obj is not None
    title = ' # '
    body = '**Test Body**'
    chunks = obj._apply_overflow(body=body, title=title)
    assert len(chunks) == 1
    assert '#\r\n**Test Body**' == chunks[0].get('body')
    assert chunks[0].get('title') == ''
    chunks = obj._apply_overflow(body=body, title=title, body_format=NotifyFormat.TEXT)
    assert len(chunks) == 1
    assert body == chunks[0].get('body')
    assert chunks[0].get('title') == ''