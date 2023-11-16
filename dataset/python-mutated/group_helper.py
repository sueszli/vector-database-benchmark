"""
Project: EverydayWechat-Github
Creator: DoubleThunder
Create time: 2019-07-11 12:55
Introduction: 群消息处理
"""
import re
from datetime import datetime
import itchat
from everyday_wechat.utils import config
from everyday_wechat.control.calendar.rt_calendar import get_rtcalendar
from everyday_wechat.utils.data_collection import get_weather_info, get_bot_info
from everyday_wechat.control.rubbish.atoolbox_rubbish import get_atoolbox_rubbish
from everyday_wechat.control.moviebox.maoyan_movie_box import get_maoyan_movie_box
from everyday_wechat.control.express.kdniao_express import get_express_info
from everyday_wechat.control.airquality.air_quality_aqicn import get_air_quality
from everyday_wechat.utils.db_helper import find_perpetual_calendar, find_user_city, find_weather, udpate_user_city, udpate_weather, update_perpetual_calendar, find_rubbish, update_rubbish, find_movie_box, update_movie_box, find_express, update_express, find_air_quality, udpate_air_quality
__all__ = ['handle_group_helper']
at_compile = '(@.*?\\s{1,}).*?'
tomorrow_compile = '明[日天]'
punct_complie = '[^a-zA-z0-9\\u4e00-\\u9fa5]+$'
help_complie = '^(?:0|帮忙|帮助|help)\\s*$'
weather_compile = '^(?:\\s*(?:1|天气|weather)(?!\\d).*?|.*?(?:天气|weather)\\s*)$'
weather_clean_compile = '1|天气|weather|\\s'
calendar_complie = '^\\s*(?:2|日历|万年历|calendar)(?=19|2[01]\\d{2}|\\s|$)'
calendar_date_compile = '^\\s*(19|2[01]\\d{2})[\\-\\/—\\s年]*(0?[1-9]|1[012])[\\-\\/—\\s月]*(0?[1-9]|[12][0-9]|3[01])[\\s日号]*$'
rubbish_complie = '^\\s*(?:3|垃圾|rubbish)(?!\\d)'
moviebox_complie = '^\\s*(?:4|票房|moviebox)(?=19|2[01]\\d{2}|\\s|$)'
express_complie = '^\\s*(?:5|快递[单号]?|express)\\s*([0-9a-zA-Z]+)'
air_compile = '^(?:\\s*(?:6|空气|pm\\s?2\\.?5)(?!\\d).*?|.*?(?:空气|pm\\s?2\\.?5)\\s*)$'
air_clean_compile = air_clean_compile = '6|空气(?:质量)?|pm\\s?2\\.?5|\\s'
common_msg = '@{ated_name}\u2005\n{text}'
weather_error_msg = '@{ated_name}\u2005\n未找到『{city}』城市的天气信息'
weather_null_msg = '@{ated_name}\u2005\n 请输入城市名'
calendar_error_msg = '@{ated_name}\u2005日期格式不对'
calendar_no_result_msg = '@{ated_name}\u2005未找到{_date}的日历数据'
rubbish_normal_msg = '@{ated_name}\u2005\n【查询结果】：『{name}』属于{_type}'
rubbish_other_msg = '@{ated_name}\u2005\n【查询结果】：『{name}』无记录\n【推荐查询】：{other}'
rubbish_nothing_msg = '@{ated_name}\u2005\n【查询结果】：『{name}』无记录'
rubbish_null_msg = '@{ated_name}\u2005 请输入垃圾名称'
moiebox_no_result_msg = '@{ated_name}\u2005未找到{_date}的票房数据'
air_city_null_msg = '@{ated_name}\u2005\n 请输入城市名'
air_error_msg = '@{ated_name}\u2005\n未找到『{city}』城市的空气质量信息'
help_group_content = '@{ated_name}\n群助手功能：\n1.输入：天气(weather)+城市名（可空）。例如：天气北京\n2.输入：日历(calendar)+日期(格式:yyyy-MM-dd 可空)。例如：日历2019-07-03\n3.输入：垃圾(rubbish)+名称。例如：3猫粮\n4.输入：票房(moviebox)+日期。例如：票房\n5.输入：快递(express)+ 快递订单号。例如: 快递 1231231231 \n6.输入：空气(pm25)+城市名。例如：pm2.5 北京\n更多功能：请输入 0|help|帮助，查看。\n'

def handle_group_helper(msg):
    if False:
        print('Hello World!')
    '\n    处理群消息\n    :param msg:\n    :return:\n    '
    uuid = msg.fromUserName
    ated_uuid = msg.actualUserName
    ated_name = msg.actualNickName
    text = msg['Text']
    if ated_uuid == config.get('wechat_uuid'):
        return
    conf = config.get('group_helper_conf')
    if not conf.get('is_open'):
        return
    if conf.get('is_at') and (not msg.isAt):
        return
    is_all = conf.get('is_all', False)
    user_uuids = conf.get('group_black_uuids') if is_all else conf.get('group_white_uuids')
    if is_all and uuid in user_uuids:
        return
    if not is_all and uuid not in user_uuids:
        return
    text = re.sub(at_compile, '', text)
    helps = re.findall(help_complie, text, re.I)
    if helps:
        retext = help_group_content.format(ated_name=ated_name)
        itchat.send(retext, uuid)
        return
    is_tomorrow = re.findall(tomorrow_compile, text)
    if is_tomorrow:
        is_tomorrow = True
        htext = re.sub(tomorrow_compile, '', text)
    else:
        is_tomorrow = False
        htext = text
    htext = re.sub(punct_complie, '', htext)
    if conf.get('is_weather'):
        if re.findall(weather_compile, htext, re.I):
            city = re.sub(weather_clean_compile, '', text, flags=re.IGNORECASE).strip()
            if not city:
                city = find_user_city(ated_uuid)
            if not city:
                city = get_city_by_uuid(ated_uuid)
            if not city:
                retext = weather_null_msg.format(ated_name=ated_name)
                itchat.send(retext, uuid)
                return
            _date = datetime.now().strftime('%Y-%m-%d')
            weather_info = find_weather(_date, city)
            if weather_info:
                retext = common_msg.format(ated_name=ated_name, text=weather_info)
                itchat.send(retext, uuid)
                return
            weather_info = get_weather_info(city)
            if weather_info:
                retext = common_msg.format(ated_name=ated_name, text=weather_info)
                itchat.send(retext, uuid)
                data = {'_date': _date, 'city_name': city, 'weather_info': weather_info, 'userid': ated_uuid, 'last_time': datetime.now()}
                udpate_weather(data)
                data2 = {'userid': ated_uuid, 'city_name': city, 'last_time': datetime.now()}
                udpate_user_city(data2)
                return
            else:
                retext = weather_error_msg.format(ated_name=ated_name, city=city)
                itchat.send(retext, uuid)
                return
            return
    if conf.get('is_calendar'):
        if re.findall(calendar_complie, htext, flags=re.IGNORECASE):
            calendar_text = re.sub(calendar_complie, '', htext).strip()
            if calendar_text:
                dates = re.findall(calendar_date_compile, calendar_text)
                if not dates:
                    retext = calendar_error_msg.format(ated_name=ated_name)
                    itchat.send(retext, uuid)
                    return
                _date = '{}-{:0>2}-{:0>2}'.format(*dates[0])
                rt_date = '{}{:0>2}{:0>2}'.format(*dates[0])
            else:
                _date = datetime.now().strftime('%Y-%m-%d')
                rt_date = datetime.now().strftime('%Y%m%d')
            cale_info = find_perpetual_calendar(_date)
            if cale_info:
                retext = common_msg.format(ated_name=ated_name, text=cale_info)
                itchat.send(retext, uuid)
                return
            cale_info = get_rtcalendar(rt_date)
            if cale_info:
                retext = common_msg.format(ated_name=ated_name, text=cale_info)
                itchat.send(retext, uuid)
                update_perpetual_calendar(_date, cale_info)
                return
            else:
                retext = calendar_no_result_msg.format(ated_name=ated_name, _date=_date)
                itchat.send(retext, uuid)
            return
    if conf.get('is_rubbish'):
        if re.findall(rubbish_complie, htext, re.I):
            key = re.sub(rubbish_complie, '', htext, flags=re.IGNORECASE).strip()
            if not key:
                retext = rubbish_null_msg.format(ated_name=ated_name)
                itchat.send(retext, uuid)
                return
            _type = find_rubbish(key)
            if _type:
                retext = rubbish_normal_msg.format(ated_name=ated_name, name=key, _type=_type)
                itchat.send(retext, uuid)
                return
            (_type, return_list, other) = get_atoolbox_rubbish(key)
            if _type:
                retext = rubbish_normal_msg.format(ated_name=ated_name, name=key, _type=_type)
                itchat.send_msg(retext, uuid)
            elif other:
                retext = rubbish_other_msg.format(ated_name=ated_name, name=key, other=other)
                itchat.send_msg(retext, uuid)
            else:
                retext = rubbish_nothing_msg.format(ated_name=ated_name, name=key)
                itchat.send_msg(retext, uuid)
            if return_list:
                update_rubbish(return_list)
            return
    if conf.get('is_moviebox'):
        if re.findall(moviebox_complie, htext, re.I):
            moviebox_text = re.sub(moviebox_complie, '', htext).strip()
            if moviebox_text:
                dates = re.findall(calendar_date_compile, moviebox_text)
                if not dates:
                    retext = calendar_error_msg.format(ated_name=ated_name)
                    itchat.send(retext, uuid)
                    return
                _date = '{}{:0>2}{:0>2}'.format(*dates[0])
            else:
                _date = datetime.now().strftime('%Y%m%d')
            mb_info = find_movie_box(_date)
            if mb_info:
                retext = common_msg.format(ated_name=ated_name, text=mb_info)
                itchat.send(retext, uuid)
                return
            is_expired = False
            cur_date = datetime.now().date()
            query_date = datetime.strptime(_date, '%Y%m%d').date()
            if query_date < cur_date:
                is_expired = True
            mb_info = get_maoyan_movie_box(_date, is_expired)
            if mb_info:
                retext = common_msg.format(ated_name=ated_name, text=mb_info)
                itchat.send(retext, uuid)
                update_movie_box(_date, mb_info, is_expired)
                return
            else:
                retext = moiebox_no_result_msg.format(ated_name=ated_name, _date=_date)
                itchat.send(retext, uuid)
            return
    if conf.get('is_express'):
        express_list = re.findall(express_complie, htext, re.I)
        if express_list:
            express_code = express_list[0]
            db_data = find_express(express_code, uuid)
            (shipper_code, shipper_name) = ('', '')
            if db_data:
                if not db_data['is_forced_update']:
                    info = db_data['info']
                    retext = common_msg.format(ated_name=ated_name, text=info)
                    itchat.send(retext, uuid)
                    return
                shipper_code = db_data['shipper_code']
                shipper_name = db_data['shipper_name']
            data = get_express_info(express_code, shipper_name=shipper_name, shipper_code=shipper_code)
            if data:
                info = data['info']
                retext = common_msg.format(ated_name=ated_name, text=info)
                itchat.send(retext, uuid)
                update_express(data, uuid)
                return
            else:
                print('未查询到此订单号的快递物流轨迹。')
                return
    if conf.get('is_air_quality'):
        if re.findall(air_compile, htext, re.I):
            city = re.sub(air_clean_compile, '', text, flags=re.IGNORECASE).strip()
            if not city:
                city = find_user_city(ated_uuid)
            if not city:
                city = get_city_by_uuid(ated_uuid)
            if not city:
                retext = air_city_null_msg.format(ated_name=ated_name)
                itchat.send(retext, uuid)
                return
            info = find_air_quality(city)
            if info:
                retext = common_msg.format(ated_name=ated_name, text=info)
                itchat.send(retext, uuid)
                return
            info = get_air_quality(city)
            if info:
                retext = common_msg.format(ated_name=ated_name, text=info)
                itchat.send(retext, uuid)
                udpate_air_quality(city, info)
                data2 = {'userid': ated_uuid, 'city_name': city, 'last_time': datetime.now()}
                udpate_user_city(data2)
                return
            else:
                retext = air_error_msg.format(ated_name=ated_name, city=city)
                itchat.send(retext, uuid)
                return
            return
    if conf.get('is_auto_reply'):
        reply_text = get_bot_info(text, ated_uuid)
        if reply_text:
            reply_text = common_msg.format(ated_name=ated_name, text=reply_text)
            itchat.send(reply_text, uuid)
            print('回复{}：{}'.format(ated_name, reply_text))
        else:
            print('自动回复失败\n')

def get_city_by_uuid(uid):
    if False:
        for i in range(10):
            print('nop')
    '\n    通过用户的uid得到用户的城市\n    最好是与机器人是好友关系\n    '
    itchat.get_friends(update=True)
    info = itchat.search_friends(userName=uid)
    if not info:
        return None
    city = info.city
    return city