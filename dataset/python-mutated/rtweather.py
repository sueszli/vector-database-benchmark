"""
https://github.com/MZCretin/RollToolsApi#获取特定城市今日天气
获取特定城市今日天气
"""
import requests
__all__ = ['get_rttodayweather']

def get_rttodayweather(cityname):
    if False:
        print('Hello World!')
    '\n    获取特定城市今日天气\n     https://github.com/MZCretin/RollToolsApi#获取特定城市今日天气\n    :param cityname:str 传入你需要查询的城市，请尽量传入完整值，否则系统会自行匹配，可能会有误差\n    :return:str 天气(2019-06-12 星期三 晴 南风 3-4级 高温 22.0℃ 低温 18.0℃ 愿你拥有比阳光明媚的心情)\n    '
    print('获取 {} 的天气...'.format(cityname))
    try:
        resp = requests.get('https://www.mxnzp.com/api/weather/forecast/{}'.format(cityname))
        print(resp.text)
        if resp.status_code == 200:
            content_dict = resp.json()
            if content_dict['code'] == 1:
                data_dict = content_dict['data']
                address = data_dict['address'].strip()
                report_time = data_dict['reportTime'].strip()
                report_time = report_time.split(' ')[0]
                return_text = ' '.join((x for x in [report_time, address, data_dict['weather'], data_dict['temp'], data_dict['windDirection'] + '风', data_dict['windPower'], '湿度：' + data_dict['humidity']] if x))
                return return_text
            else:
                print('获取天气失败:{}'.format(content_dict['msg']))
        print('获取天气失败。')
    except Exception as exception:
        print(str(exception))
get_today_weather = get_rttodayweather
if __name__ == '__main__':
    cityname = '香港'
    weather = get_today_weather(cityname)
    pass