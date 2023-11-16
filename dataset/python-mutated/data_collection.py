"""
获取各种请求的调度管理中心
"""
import importlib
import re
from datetime import datetime
from datetime import timedelta
from everyday_wechat.control.weather.sojson import get_sojson_weather
from everyday_wechat.utils.common import get_constellation_name
from everyday_wechat.utils import config
from everyday_wechat.control.horoscope.xzw_horescope import get_today_horoscope
from everyday_wechat.control.calendar.rt_calendar import get_rtcalendar
__all__ = ['get_dictum_info', 'get_weather_info', 'get_bot_info', 'get_diff_time', 'get_constellation_info', 'get_calendar_info', 'DICTUM_NAME_DICT', 'BOT_NAME_DICT']
DICTUM_NAME_DICT = {1: 'wufazhuce', 2: 'acib', 3: 'lovelive', 4: 'hitokoto', 5: 'rtjokes', 6: 'juzimi', 7: 'caihongpi'}
BOT_NAME_DICT = {1: 'tuling123', 2: 'yigeai', 3: 'qingyunke', 4: 'qq_nlpchat', 5: 'tian_robot', 6: 'ruyiai', 7: 'ownthink_robot'}
BIRTHDAY_COMPILE = re.compile('\\-?(0?[1-9]|1[012])\\-(0?[1-9]|[12][0-9]|3[01])$')

def get_dictum_info(channel):
    if False:
        i = 10
        return i + 15
    '\n    获取每日一句。\n    :return:str\n    '
    if not channel:
        return None
    source = DICTUM_NAME_DICT.get(channel, '')
    if source:
        addon = importlib.import_module('everyday_wechat.control.onewords.' + source, __package__)
        dictum = addon.get_one_words()
        return dictum
    return None

def get_weather_info(cityname, is_tomorrow=False):
    if False:
        i = 10
        return i + 15
    '\n    获取天气\n    :param cityname:str,城市名称\n    :return: str,天气情况\n    '
    if not cityname:
        return
    return get_sojson_weather(cityname, is_tomorrow)

def get_bot_info(message, userId=''):
    if False:
        i = 10
        return i + 15
    '\n    跟机器人互动\n    # 优先获取图灵机器人API的回复，但失效时，会使用青云客智能聊天机器人API(过时)\n    :param message:str, 发送的话\n    :param userId: str, 好友的uid，作为请求的唯一标识。\n    :return:str, 机器人回复的话。\n    '
    channel = config.get('auto_reply_info').get('bot_channel', 7)
    source = BOT_NAME_DICT.get(channel, 'ownthink_robot')
    if source:
        addon = importlib.import_module('everyday_wechat.control.bot.' + source, __package__)
        reply_msg = addon.get_auto_reply(message, userId)
        return reply_msg
    return None

def get_diff_time(start_date, start_msg=''):
    if False:
        return 10
    '\n    # 在一起，一共多少天了。\n    :param start_date:str,日期\n    :return: str,eg（宝贝这是我们在一起的第 111 天。）\n    '
    if not start_date:
        return None
    rdate = '^[12]\\d{3}[ \\/\\-](?:0?[1-9]|1[012])[ \\/\\-](?:0?[1-9]|[12][0-9]|3[01])$'
    start_date = start_date.strip()
    if not re.search(rdate, start_date):
        print('日期填写出错..')
        return
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    day_delta = (datetime.now() - start_datetime).days + 1
    if start_msg and start_msg.count('{}') == 1:
        delta_msg = start_msg.format(day_delta)
    else:
        delta_msg = '宝贝这是我们在一起的第 {} 天。'.format(day_delta)
    return delta_msg

def get_constellation_info(birthday_str, is_tomorrow=False):
    if False:
        print('Hello World!')
    '\n    获取星座运势\n    :param birthday_str:  "10-12" 或  "1980-01-08" 或 星座名\n    :return:\n    '
    if not birthday_str:
        return
    const_name = get_constellation_name(birthday_str)
    if not const_name:
        print('星座名填写错误')
        return
    return get_today_horoscope(const_name, is_tomorrow)

def get_calendar_info(calendar=True, is_tomorrow=False, _date=''):
    if False:
        for i in range(10):
            print('nop')
    ' 获取万年历 '
    if not calendar:
        return None
    if not is_tomorrow:
        date = datetime.now().strftime('%Y%m%d')
    else:
        date = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
    return get_rtcalendar(date)
if __name__ == '__main__':
    config.init()
    text = 'are you ok'
    reply_msg = get_bot_info(text)
    print(reply_msg)
    pass