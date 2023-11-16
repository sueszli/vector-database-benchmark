"""
Project: EverydayWechat-Github
Creator: DoubleThunder
Create time: 2019-07-14 01:06
Introduction:
"""
from unittest import TestCase
from everyday_wechat.control.rubbish.atoolbox_rubbish import get_atoolbox_rubbish

class BaseTestCase(TestCase):

    def test_atoolbox_rubbish(self):
        if False:
            for i in range(10):
                print('nop')
        key = '牛'
        ok = get_atoolbox_rubbish(key)
        print(ok)