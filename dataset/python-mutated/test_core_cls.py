from __future__ import unicode_literals
from pypinyin.constants import Style
from pypinyin.core import Pinyin, to_fixed, handle_nopinyin, single_pinyin, phrase_pinyin

def test_use_pre_seg_to_skip_seg():
    if False:
        while True:
            i = 10

    class A(Pinyin):

        def pre_seg(self, hans, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return ['a', 'b', 'c']
    mypinyin = A()
    assert Pinyin().pinyin('测试') == [['cè'], ['shì']]
    assert mypinyin.pinyin('测试') == [['a'], ['b'], ['c']]

def test_use_post_seg_to_change_seg_result():
    if False:
        while True:
            i = 10

    class A(Pinyin):

        def post_seg(self, hans, seg_data, **kwargs):
            if False:
                return 10
            return ['a', 'b', 'c']
    mypinyin = A()
    assert Pinyin().pinyin('测试') == [['cè'], ['shì']]
    assert mypinyin.pinyin('测试') == [['a'], ['b'], ['c']]

def test_use_seg_function_change_seg_func():
    if False:
        while True:
            i = 10

    def seg(han):
        if False:
            return 10
        return ['a', 'b', 'c']

    class A(Pinyin):

        def get_seg(self):
            if False:
                print('Hello World!')
            return seg
    mypinyin = A()
    assert Pinyin().pinyin('测试') == [['cè'], ['shì']]
    assert mypinyin.pinyin('测试') == [['a'], ['b'], ['c']]

def test_to_fixed_for_compatibly():
    if False:
        i = 10
        return i + 15
    assert to_fixed('cè', Style.INITIALS) == 'c'

def test_handle_nopinyin_for_compatibly():
    if False:
        return 10
    assert handle_nopinyin('test') == [['test']]

def test_single_pinyin_for_compatibly():
    if False:
        print('Hello World!')
    assert single_pinyin('测', Style.TONE, False) == [['cè']]

def test_phrase_pinyin_for_compatibly():
    if False:
        while True:
            i = 10
    assert phrase_pinyin('测试', Style.TONE, False) == [['cè'], ['shì']]