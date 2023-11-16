"""
Project: EverydayWechat-Github
Creator: DoubleThunder
Create time: 2019年9月4日12:41:33
Introduction:
"""
from tests import BaseTestCase
from everyday_wechat.utils.db_helper import *
from everyday_wechat.control.airquality.air_quality_aqicn import get_air_quality
import pysnooper

class TestDbAirModel(BaseTestCase):

    def test_db_get_data(self):
        if False:
            while True:
                i = 10
        city = '天津'
        cc = get_air_quality(city)
        if cc:
            print(cc)

    def test_db_get_and_save_data(self):
        if False:
            print('Hello World!')
        city = '天津'
        info = get_air_quality(city)
        if info:
            udpate_air_quality(city, info)

    def test_db_find_data(self):
        if False:
            i = 10
            return i + 15
        city = '天津'
        info = find_air_quality(city)
        if info:
            print(info)

    @pysnooper.snoop()
    def test_all_data(self):
        if False:
            print('Hello World!')
        city = '宁波'
        info = find_air_quality(city)
        if info:
            print(info)
            return
        info = get_air_quality(city)
        if info:
            print(info)
            udpate_air_quality(city, info)