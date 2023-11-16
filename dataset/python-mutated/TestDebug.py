from Debug import Debug
import gevent
import os
import re
import pytest

class TestDebug:

    @pytest.mark.parametrize('items,expected', [(['@/src/A/B/C.py:17'], ['A/B/C.py line 17']), (['@/src/Db/Db.py:17'], ['Db.py line 17']), (['%s:1' % __file__], ['TestDebug.py line 1']), (['@/plugins/Chart/ChartDb.py:100'], ['ChartDb.py line 100']), (['@/main.py:17'], ['main.py line 17']), (['@\\src\\Db\\__init__.py:17'], ['Db/__init__.py line 17']), (['<frozen importlib._bootstrap>:1'], []), (['<frozen importlib._bootstrap_external>:1'], []), (['/home/ivanq/ZeroNet/src/main.py:13'], ['?/src/main.py line 13']), (['C:\\ZeroNet\\core\\src\\main.py:13'], ['?/src/main.py line 13']), (['/root/main.py:17'], ['/root/main.py line 17']), (['{gevent}:13'], ['<gevent>/__init__.py line 13']), (['{os}:13'], ['<os> line 13']), (['src/gevent/event.py:17'], ['<gevent>/event.py line 17']), (['@/src/Db/Db.py:17', '@/src/Db/DbQuery.py:1'], ['Db.py line 17', 'DbQuery.py line 1']), (['@/src/Db/Db.py:17', '@/src/Db/Db.py:1'], ['Db.py line 17', '1']), (['{os}:1', '@/src/Db/Db.py:17'], ['<os> line 1', 'Db.py line 17']), (['{gevent}:1'] + ['{os}:3'] * 4 + ['@/src/Db/Db.py:17'], ['<gevent>/__init__.py line 1', '...', 'Db.py line 17'])])
    def testFormatTraceback(self, items, expected):
        if False:
            for i in range(10):
                print('nop')
        q_items = []
        for item in items:
            (file, line) = item.rsplit(':', 1)
            if file.startswith('@'):
                file = Debug.root_dir + file[1:]
            file = file.replace('{os}', os.__file__)
            file = file.replace('{gevent}', gevent.__file__)
            q_items.append((file, int(line)))
        assert Debug.formatTraceback(q_items) == expected

    def testFormatException(self):
        if False:
            while True:
                i = 10
        try:
            raise ValueError('Test exception')
        except Exception:
            assert re.match('ValueError: Test exception in TestDebug.py line [0-9]+', Debug.formatException())
        try:
            os.path.abspath(1)
        except Exception:
            assert re.search('in TestDebug.py line [0-9]+ > <(posixpath|ntpath)> line ', Debug.formatException())

    def testFormatStack(self):
        if False:
            return 10
        assert re.match('TestDebug.py line [0-9]+ > <_pytest>/python.py line [0-9]+', Debug.formatStack())