from h2o_wave import Expando, expando_to_dict, clone_expando, copy_expando
import unittest

class TestExpando(unittest.TestCase):

    def test_expando_empty(self):
        if False:
            return 10
        e = Expando()
        d = expando_to_dict(e)
        assert len(d) == 0

    def test_expando_create(self):
        if False:
            return 10
        e = Expando(dict(answer=42))
        assert e.answer == 42
        assert e['answer'] == 42
        assert 'answer' in e
        d = expando_to_dict(e)
        assert len(d) == 1
        assert d['answer'] == 42

    def test_expando_dict_write(self):
        if False:
            i = 10
            return i + 15
        e = Expando(dict(answer=42))
        d = expando_to_dict(e)
        d['answer'] = 43
        assert e['answer'] == 43

    def test_expando_item_write(self):
        if False:
            while True:
                i = 10
        e = Expando()
        assert e.answer is None
        assert e['answer'] is None
        assert 'answer' not in e
        e['answer'] = 42
        assert e.answer == 42
        assert e['answer'] == 42
        assert 'answer' in e

    def test_expando_attr_write(self):
        if False:
            while True:
                i = 10
        e = Expando()
        assert e.answer is None
        assert e['answer'] is None
        assert 'answer' not in e
        e.answer = 42
        assert e.answer == 42
        assert e['answer'] == 42
        assert 'answer' in e

    def test_expando_item_del(self):
        if False:
            i = 10
            return i + 15
        e = Expando(dict(answer=42))
        assert e.answer == 42
        assert e['answer'] == 42
        assert 'answer' in e
        del e['answer']
        assert e.answer is None
        assert e['answer'] is None
        assert 'answer' not in e

    def test_expando_attr_del(self):
        if False:
            return 10
        e = Expando(dict(answer=42))
        assert e.answer == 42
        assert e['answer'] == 42
        assert 'answer' in e
        del e.answer
        assert e.answer is None
        assert e['answer'] is None
        assert 'answer' not in e

    def test_expando_clone(self):
        if False:
            i = 10
            return i + 15
        e = clone_expando(Expando(dict(answer=42)))
        assert e.answer == 42
        assert e['answer'] == 42
        assert 'answer' in e

    def test_expando_copy(self):
        if False:
            print('Hello World!')
        e = copy_expando(Expando(dict(answer=42)), Expando())
        assert e.answer == 42
        assert e['answer'] == 42
        assert 'answer' in e

    def test_expando_clone_exclude_keys(self):
        if False:
            i = 10
            return i + 15
        e = clone_expando(Expando(dict(a=1, b=2, c=3)), exclude_keys=['a'])
        assert 'a' not in e
        assert 'b' in e
        assert 'c' in e

    def test_expando_clone_include_keys(self):
        if False:
            return 10
        e = clone_expando(Expando(dict(a=1, b=2, c=3)), include_keys=['a', 'b'])
        assert 'a' in e
        assert 'b' in e
        assert 'c' not in e

    def test_expando_clone_include_exclued_keys(self):
        if False:
            for i in range(10):
                print('nop')
        e = clone_expando(Expando(dict(a=1, b=2, c=3)), include_keys=['a', 'b'], exclude_keys=['b', 'c'])
        assert 'a' in e
        assert 'b' not in e
        assert 'c' not in e