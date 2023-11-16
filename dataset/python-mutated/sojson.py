import requests
import json
import os
from datetime import datetime
from datetime import timedelta
__all__ = ['get_sojson_weather', 'get_sojson_weather_tomorrow']
with open(os.path.join(os.path.dirname(__file__), '_city_sojson.json'), 'r', encoding='utf-8') as f:
    CITY_CODE_DICT = json.loads(f.read())
MSG_TOMORROW = '明日{city_name}天气\n{_date} {week}\n【明日天气】{_type}\n【明日气温】{low_temp} {high_temp}\n【明日风速】{speed}\n【出行提醒】{notice}'
MSG_TODAY = '今日{city_name}天气\n{_date},{week}\n【今日天气】{_type}\n【今日气温】{low_temp} {high_temp}\n【今日风速】{speed}\n【出行提醒】{notice}'

def get_sojson_weather(city_name, is_tomorrow=False):
    if False:
        while True:
            i = 10
    '\n     获取天气信息。网址：https://www.sojson.com/blog/305.html .\n    :param city_name: str,城市名\n    :return: str ,例如：2019-06-12 星期三 晴 南风 3-4级 高温 22.0℃ 低温 18.0℃ 愿你拥有比阳光明媚的心情\n    '
    if is_tomorrow:
        return get_sojson_weather_tomorrow(city_name)
    if not city_name:
        return None
    city_code = CITY_CODE_DICT.get(city_name, None)
    if not city_code:
        print('没有此城市的消息...')
        return None
    print('获取天气信息...')
    weather_url = 'http://t.weather.sojson.com/api/weather/city/{}'.format(city_code)
    try:
        resp = requests.get(url=weather_url)
        if resp.status_code == 200:
            weather_dict = resp.json()
            if weather_dict.get('status') == 200:
                today_weather = weather_dict.get('data').get('forecast')[0]
                today_date = datetime.now().strftime('%Y-%m-%d')
                weather_today = today_weather['ymd']
                if today_date != weather_today:
                    today_weather = weather_dict.get('data').get('forecast')[1]
                weather_info = MSG_TODAY.format(city_name=city_name, _date=today_weather['ymd'], week=today_weather['week'], _type=today_weather['type'], low_temp=today_weather['low'], high_temp=today_weather['high'], speed=today_weather['fx'] + today_weather['fl'], notice=today_weather['notice'])
                return weather_info
            else:
                print('天气请求出错:{}'.format(weather_dict.get('message')))
    except Exception as exception:
        print(str(exception))
        return None

def get_sojson_weather_tomorrow(city_name):
    if False:
        return 10
    '\n     获取明日天气信息。网址：https://www.sojson.com/blog/305.html .\n    :param city_name: str,城市名\n    :return: str ,例如：2019-06-12 星期三 晴 南风 3-4级 高温 22.0℃ 低温 18.0℃ 愿你拥有比阳光明媚的心情\n    '
    if not city_name:
        return None
    city_code = CITY_CODE_DICT.get(city_name, None)
    if not city_code:
        print('没有此城市的消息...')
        return None
    print('获取天气信息...')
    weather_url = 'http://t.weather.sojson.com/api/weather/city/{}'.format(city_code)
    try:
        resp = requests.get(url=weather_url)
        if resp.status_code == 200:
            weather_dict = resp.json()
            if weather_dict.get('status') == 200:
                today_weather = weather_dict.get('data').get('forecast')[1]
                today_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                weather_today = today_weather['ymd']
                if today_date != weather_today:
                    today_weather = weather_dict.get('data').get('forecast')[2]
                weather_info = MSG_TOMORROW.format(city_name=city_name, _date=today_weather['ymd'], week=today_weather['week'], _type=today_weather['type'], low_temp=today_weather['low'], high_temp=today_weather['high'], speed=today_weather['fx'] + today_weather['fl'], notice=today_weather['notice'])
                return weather_info
            else:
                print('天气请求出错:{}'.format(weather_dict.get('message')))
    except Exception as exception:
        print(str(exception))
        return None
get_today_weather = get_sojson_weather
if __name__ == '__main__':
    is_tomorrow = True
    we = get_sojson_weather('青岛', is_tomorrow)
    print(we)