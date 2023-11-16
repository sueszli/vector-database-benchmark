import pytest
from celery.utils.text import abbr, abbrtask, ensure_newlines, indent, pretty, truncate
RANDTEXT = 'The quick brown\nfox jumps\nover the\nlazy dog'
RANDTEXT_RES = '    The quick brown\n    fox jumps\n    over the\n    lazy dog'
QUEUES = {'queue1': {'exchange': 'exchange1', 'exchange_type': 'type1', 'routing_key': 'bind1'}, 'queue2': {'exchange': 'exchange2', 'exchange_type': 'type2', 'routing_key': 'bind2'}}
QUEUE_FORMAT1 = '.> queue1           exchange=exchange1(type1) key=bind1'
QUEUE_FORMAT2 = '.> queue2           exchange=exchange2(type2) key=bind2'

class test_Info:

    def test_textindent(self):
        if False:
            print('Hello World!')
        assert indent(RANDTEXT, 4) == RANDTEXT_RES

    def test_format_queues(self, app):
        if False:
            i = 10
            return i + 15
        app.amqp.queues = app.amqp.Queues(QUEUES)
        assert sorted(app.amqp.queues.format().split('\n')) == sorted([QUEUE_FORMAT1, QUEUE_FORMAT2])

    def test_ensure_newlines(self):
        if False:
            while True:
                i = 10
        assert len(ensure_newlines('foo\nbar\nbaz\n').splitlines()) == 3
        assert len(ensure_newlines('foo\nbar').splitlines()) == 2

@pytest.mark.parametrize('s,maxsize,expected', [('ABCDEFGHI', 3, 'ABC...'), ('ABCDEFGHI', 10, 'ABCDEFGHI')])
def test_truncate_text(s, maxsize, expected):
    if False:
        for i in range(10):
            print('nop')
    assert truncate(s, maxsize) == expected

@pytest.mark.parametrize('args,expected', [((None, 3), '???'), (('ABCDEFGHI', 6), 'ABC...'), (('ABCDEFGHI', 20), 'ABCDEFGHI'), (('ABCDEFGHI', 6, None), 'ABCDEF')])
def test_abbr(args, expected):
    if False:
        i = 10
        return i + 15
    assert abbr(*args) == expected

@pytest.mark.parametrize('s,maxsize,expected', [(None, 3, '???'), ('feeds.tasks.refresh', 10, '[.]refresh'), ('feeds.tasks.refresh', 30, 'feeds.tasks.refresh')])
def test_abbrtask(s, maxsize, expected):
    if False:
        print('Hello World!')
    assert abbrtask(s, maxsize) == expected

def test_pretty():
    if False:
        while True:
            i = 10
    assert pretty(('a', 'b', 'c'))