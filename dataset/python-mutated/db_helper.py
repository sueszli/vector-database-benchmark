"""
Project: EverydayWechat-Github
Creator: DoubleThunder
Create time: 2019-07-12 18:37
Introduction:
"""
import pymongo
from everyday_wechat.utils import config
from functools import wraps
from datetime import datetime
__all__ = ['is_open_db', 'udpate_weather', 'udpate_user_city', 'find_user_city', 'find_weather', 'update_perpetual_calendar', 'find_perpetual_calendar', 'find_rubbish', 'update_rubbish', 'find_movie_box', 'update_movie_box', 'find_express', 'update_express', 'find_air_quality', 'udpate_air_quality']
cache_valid_time = 4 * 60 * 60
db_config = config.get('db_config')
if db_config and db_config.get('is_open_db') and db_config.get('mongodb_conf'):
    is_open_db = db_config.get('is_open_db')
    mongodb_conf = db_config.get('mongodb_conf')
    try:
        myclient = pymongo.MongoClient(host=mongodb_conf.get('host'), port=mongodb_conf.get('port'), serverSelectionTimeoutMS=10)
        myclient.server_info()
        wechat_helper_db = myclient['wechat_helper']
        weather_db = wechat_helper_db['weather']
        user_city_db = wechat_helper_db['user_city']
        perpetual_calendar_db = wechat_helper_db['perpetual_calendar']
        rubbish_db = wechat_helper_db['rubbish_assort']
        movie_box_db = wechat_helper_db['movie_box']
        express_db = wechat_helper_db['express']
        air_quality_db = wechat_helper_db['air_quality']
    except pymongo.errors.ServerSelectionTimeoutError as err:
        is_open_db = False
else:
    is_open_db = False

def db_flag():
    if False:
        while True:
            i = 10
    ' 用于数据库操作的 flag 没开启就不进行数据库操作'

    def _db_flag(func):
        if False:
            return 10

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            if is_open_db:
                return func(*args, **kwargs)
            else:
                return None
        return wrapper
    return _db_flag

@db_flag()
def udpate_weather(data):
    if False:
        while True:
            i = 10
    '\n    更新天气数据\n    :param data:\n    '
    key = {'_date': data['_date'], 'city_name': data['city_name']}
    weather_db.update_one(key, {'$set': data}, upsert=True)

@db_flag()
def udpate_user_city(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    更新用户城市信息，用户最后一次查询成功的城市名\n    :param data:\n    '
    key = {'userid': data['userid']}
    user_city_db.update_one(key, {'$set': data}, upsert=True)

@db_flag()
def find_user_city(uuid):
    if False:
        while True:
            i = 10
    '\n    找到用户的城市，用户最后一次查询的城市名\n    :param uuid:\n    :return:\n    '
    key = {'userid': uuid}
    data = user_city_db.find_one(key)
    if data:
        return data['city_name']

@db_flag()
def find_weather(date, cityname):
    if False:
        while True:
            i = 10
    '\n    根据日期与城市名获取天气信息，天气信息有效期为 4 小时\n    :param date: 日期(yyyy-mm-dd)\n    :param cityname: 城市名\n    :return: 天气信息\n    '
    key = {'_date': date, 'city_name': cityname}
    data = weather_db.find_one(key)
    if data:
        diff_second = (datetime.now() - data['last_time']).seconds
        if diff_second <= cache_valid_time:
            return data['weather_info']
    return None

@db_flag()
def update_perpetual_calendar(_date, info):
    if False:
        i = 10
        return i + 15
    '\n    更新日历信息\n    :param _date: 日期(yyyy-mm-dd)\n    :param info: 内容\n    :return: None\n    '
    key = {'_date': _date}
    data = {'_date': _date, 'info': info, 'last_time': datetime.now()}
    perpetual_calendar_db.update_one(key, {'$set': data}, upsert=True)

@db_flag()
def find_perpetual_calendar(_date):
    if False:
        for i in range(10):
            print('nop')
    '\n    查找日历内容\n    :param _date: str 日期(yyyy-mm-dd)\n    :return: str\n    '
    key = {'_date': _date}
    data = perpetual_calendar_db.find_one(key)
    if data:
        return data['info']

@db_flag()
def find_rubbish(name):
    if False:
        i = 10
        return i + 15
    "\n    从数据库里查询获取内容\n    {'name': '爱群主', 'type': '什么垃圾'}\n    "
    key = {'name': name}
    one = rubbish_db.find_one(key, {'_id': 0, 'name': 1, 'type': 1})
    if one:
        return one['type']
    return None

@db_flag()
def update_rubbish(data):
    if False:
        while True:
            i = 10
    '\n    将垃圾保存数据库\n    :param data:\n    :return:\n    '
    if isinstance(data, str):
        data = [data]
    if isinstance(data, list):
        for d in data:
            key = {'name': d['name']}
            value = {'$set': {'type': d['type']}}
            rubbish_db.update_one(key, value, upsert=True)

@db_flag()
def find_movie_box(date):
    if False:
        print('Hello World!')
    '\n    查询电脑票房，\n    如果是历史票房，则直接返回数据\n    如果不是，保存时间在5分钟内，则直接返回数据。\n    其他情况，返回为空\n    :param date: 查询时间\n    :return:\n    '
    key = {'_date': date}
    data = movie_box_db.find_one(key)
    if data:
        is_expired = data['is_expired']
        if is_expired:
            return data['info']
        diff_second = (datetime.now() - data['last_time']).seconds
        if diff_second <= 5 * 60:
            return data['info']
    return None

@db_flag()
def update_movie_box(date, info, is_expired=False):
    if False:
        i = 10
        return i + 15
    '\n    保存实时票房\n    :param date: 日期 yyyyDDmm 格式\n    :param info: 票房内容\n    :param is_today: 是否是今日实时票房\n    :return: None\n    '
    key = {'_date': date}
    data = {'_date': date, 'info': info, 'last_time': datetime.now(), 'is_expired': is_expired}
    movie_box_db.update_one(key, {'$set': data}, upsert=True)

@db_flag()
def update_express(data, uuid):
    if False:
        print('Hello World!')
    "\n    更新快递内容, 包括\n    {'express_code': '78109182715352','shipper_code': 'ZTO',\n    'shipper_name': '中通速递','info': '很多内容', 'state': True}\n    :param data: dict 内容数据\n    :param uuid: str 用户 uid\n    :return:\n    "
    key = {'express_code': data['express_code']}
    data['userid'] = uuid
    data['last_time'] = datetime.now()
    express_db.update_one(key, {'$set': data}, upsert=True)
    return None

@db_flag()
def find_express(express_code='', uuid=''):
    if False:
        print('Hello World!')
    '\n    获取缓存快递信息，express_code ,uuid 不可同时为空\n    缓存时间：5 分钟\n    :param express_code: str,快递单号\n    :param uuid: str,用户 uid\n    :return: dict ,快递信息\n    '
    key = {}
    if express_code:
        key['express_code'] = express_code
    elif uuid:
        key['userid'] = uuid
    else:
        return None
    data = express_db.find_one(key)
    if data:
        data['is_forced_update'] = False
        state = data['state']
        if state:
            return data
        diff_second = (datetime.now() - data['last_time']).seconds
        if diff_second <= 5 * 60:
            return data
        else:
            data['is_forced_update'] = True
            return data
    return None

@db_flag()
def find_air_quality(city):
    if False:
        while True:
            i = 10
    '\n    根据日期与城市名获取空气信息，pm2.5 记录有效期为 1 小时\n    :param city: 城市名\n    :return: 空气信息\n    '
    key = {'city': city}
    data = air_quality_db.find_one(key)
    if data:
        diff_second = (datetime.now() - data['last_time']).seconds
        if diff_second <= 1 * 60 * 60:
            return data['info']
    return None

@db_flag()
def udpate_air_quality(city, info):
    if False:
        return 10
    '\n    :param city: 城市名\n    :param info: 空气情况\n    :return:\n    '
    key = {'city': city}
    data = {'city': city, 'info': info, 'last_time': datetime.now()}
    air_quality_db.update_one(key, {'$set': data}, upsert=True)