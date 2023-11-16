"""
Project: EverydayWechat-Github
Creator: DoubleThunder
Create time: 2019-07-11 14:56
Introduction:
"""
import itchat
import re
from datetime import datetime
from datetime import timedelta
from importlib import import_module
from everyday_wechat.utils import config
from everyday_wechat.utils.common import md5_encode, FILEHELPER_MARK, FILEHELPER
from everyday_wechat.utils.data_collection import BOT_NAME_DICT
__all__ = ['init_wechat_config', 'set_system_notice', 'get_group', 'get_friend']
TIME_COMPILE = re.compile('^\\s*([01]?[0-9]|2[0-3])\\s*[：:\\-]\\s*([0-5]?[0-9])\\s*$')

def init_wechat_config():
    if False:
        while True:
            i = 10
    ' 初始化微信所需数据 '
    myset = config.copy()
    print('=' * 80)
    base_wechat_info = itchat.search_friends()
    wechat_nick_name = base_wechat_info['NickName']
    wechat_uuid = base_wechat_info['UserName']
    myset['wechat_nick_name'] = wechat_nick_name
    myset['wechat_uuid'] = wechat_uuid
    reply = myset.get('auto_reply_info')
    if reply is not None and reply.get('is_auto_reply'):
        if reply.get('is_auto_reply_all'):
            auto_reply_list_key = 'auto_reply_black_list'
            auto_reply_list_uuid_name = 'auto_reply_black_uuids'
        else:
            auto_reply_list_key = 'auto_reply_white_list'
            auto_reply_list_uuid_name = 'auto_reply_white_uuids'
        auto_reply_uuids_list = []
        for name in reply.get(auto_reply_list_key):
            if not name.strip():
                continue
            if name.lower() in FILEHELPER_MARK:
                auto_reply_uuids_list.append(FILEHELPER)
                continue
            friend = get_friend(name)
            if friend:
                auto_reply_uuids_list.append(friend['UserName'])
            else:
                print('自动回复中的好友昵称『{}』有误。'.format(name))
        reply[auto_reply_list_uuid_name] = set(auto_reply_uuids_list)
    helper = myset.get('group_helper_conf')
    if helper is not None and helper.get('is_open'):
        if helper.get('is_all', False):
            group_list_key = 'group_name_black_list'
            group_list_uuid_name = 'group_black_uuids'
        else:
            group_list_key = 'group_name_white_list'
            group_list_uuid_name = 'group_white_uuids'
        group_uuid_list = []
        for name in helper.get(group_list_key):
            if not name.strip():
                continue
            group = get_group(name)
            if group:
                group_uuid_list.append(group['UserName'])
            else:
                print('群助手中的群聊名称『{}』有误。(注意：必须要把需要的群聊保存到通讯录)'.format(name))
        helper[group_list_uuid_name] = set(group_uuid_list)
    alarm = myset.get('alarm_info')
    alarm_dict = {}
    if alarm is not None and alarm.get('is_alarm'):
        for gi in alarm.get('girlfriend_infos'):
            ats = gi.get('alarm_timed')
            if not ats:
                continue
            uuid_list = []
            nickname_list = []
            friends = gi.get('wechat_name')
            if isinstance(friends, str):
                friends = [friends]
            if isinstance(friends, list):
                for name in friends:
                    if name.lower() in FILEHELPER_MARK:
                        uuid_list.append(FILEHELPER)
                        nickname_list.append(name)
                        continue
                    name_info = get_friend(name)
                    if not name_info:
                        print('定时提醒中的好友昵称『{}』无效'.format(name))
                    else:
                        uuid_list.append(name_info['UserName'])
                        nickname_list.append(name)
            group_names = gi.get('group_name')
            if isinstance(group_names, str):
                group_names = [group_names]
            if isinstance(group_names, list):
                for name in group_names:
                    name_info = get_group(name)
                    if not name_info:
                        print('定时任务中的群聊名称『{}』有误。(注意：必须要把需要的群聊保存到通讯录)'.format(name))
                    else:
                        uuid_list.append(name_info['UserName'])
                        nickname_list.append(name)
            if isinstance(ats, str):
                ats = [ats]
            if isinstance(ats, list):
                for at in ats:
                    times = TIME_COMPILE.findall(at)
                    if not times:
                        print('时间{}格式出错'.format(at))
                        continue
                    (hour, minute) = (int(times[0][0]), int(times[0][1]))
                    temp_dict = {'hour': hour, 'minute': minute, 'uuid_list': uuid_list, 'nickname_list': nickname_list}
                    temp_dict.update(gi)
                    alarm_dict[md5_encode(str(temp_dict))] = temp_dict
        alarm['alarm_dict'] = alarm_dict
    config.update(myset)
    log_all_config()

def set_system_notice(text):
    if False:
        print('Hello World!')
    '\n    给文件传输助手发送系统日志。\n    :param text:str 日志内容\n    '
    if text:
        text = '系统通知：' + text
        itchat.send(text, toUserName=FILEHELPER)

def get_group(group_name, update=False):
    if False:
        print('Hello World!')
    '\n    根据群组名获取群组数据\n    :param group_name:str, 群组名\n    :param update: bool 强制更新群组数据\n    :return: obj 单个群组信息\n    '
    if update:
        itchat.get_chatrooms(update=True)
    if not group_name:
        return None
    groups = itchat.search_chatrooms(name=group_name)
    if not groups:
        return None
    return groups[0]

def get_friend(wechat_name, update=False):
    if False:
        while True:
            i = 10
    '\n    根据用户名获取用户数据\n    :param wechat_name: str 用户名\n    :param update: bool 强制更新用户数据\n    :return: obj 单个好友信息\n    '
    if update:
        itchat.get_friends(update=True)
    if not wechat_name:
        return None
    friends = itchat.search_friends(name=wechat_name)
    if not friends:
        return None
    return friends[0]

def get_mps(mp_name, update=False):
    if False:
        print('Hello World!')
    '\n    根据公众号的名称获取用户数据\n    :param mp_name: str 用户名\n    :param update: bool 强制更新用户数据\n    :return: obj 单个公众号信息\n    '
    if update:
        itchat.get_mps(update=True)
    if not mp_name:
        return None
    mps = itchat.search_mps(name=mp_name)
    if not mps:
        return None
    return mps[0]

def log_all_config():
    if False:
        return 10
    '\n    用于打印设置日志\n    :return:\n    '
    print('=' * 80)
    channel = config.get('auto_reply_info').get('bot_channel', 7)
    source = BOT_NAME_DICT.get(channel, 'ownthink_robot')
    addon = import_module('everyday_wechat.control.bot.' + source, __package__)
    bot_name = addon.BOT_NAME
    print('自动回复机器人渠道：{}'.format(bot_name))
    reply = config.get('auto_reply_info', None)
    if not reply or not reply.get('is_auto_reply'):
        print('未开启微信好友自动回复。')
    elif reply.get('is_auto_reply_all'):
        auto_uuids = reply.get('auto_reply_black_uuids')
        nicknames = []
        for auid in auto_uuids:
            if auid == 'filehelper':
                nicknames.append(auid)
            else:
                friends = itchat.search_friends(userName=auid)
                nickname = friends.nickName
                nicknames.append(nickname)
        nns = '，'.join(nicknames)
        print('开启对全部微信好友全部回复，除了：{}'.format(nns))
    else:
        auto_uuids = reply.get('auto_reply_white_uuids')
        nicknames = []
        for auid in auto_uuids:
            if auid == 'filehelper':
                nicknames.append(auid)
            else:
                friends = itchat.search_friends(userName=auid)
                nickname = friends.nickName
                nicknames.append(nickname)
        nns = '，'.join(nicknames)
        print('对微信好友 {}，进行自动回复'.format(nns))
    print('=' * 80)
    helper = config.get('group_helper_conf')
    if not helper or not helper.get('is_open'):
        print('未开启群助手功能。')
    elif helper.get('is_all'):
        auto_uuids = helper.get('group_black_uuids')
        nicknames = []
        for auid in auto_uuids:
            chatrooms = itchat.search_chatrooms(userName=auid)
            nickname = chatrooms['NickName']
            nicknames.append(nickname)
        nns = '，'.join(nicknames)
        print('已开启对全部微信群的监听，除了群：{}。'.format(nns))
    else:
        auto_uuids = helper.get('group_white_uuids')
        nicknames = []
        for auid in auto_uuids:
            chatroom = itchat.search_chatrooms(userName=auid)
            nickname = chatroom['NickName']
            nicknames.append(nickname)
        nns = '，'.join(nicknames)
        print('已对微信群：{}，开启了群助手功能。'.format(nns))
        if helper.get('is_at'):
            print('只有群里用户@机器人，才会触发群助手功能。')
        if helper.get('is_auto_reply'):
            print('已开启对微信群内用户的自动回复。')
        if helper.get('is_weather'):
            print('已开启天气查询功能，具体使用方法请输入：“help” 查看。')
        if helper.get('is_calendar'):
            print('已开启日志查询功能，具体使用方法请输入：“help” 查看。')
        if helper.get('is_rubbish'):
            print('已开启垃圾分类查询功能，具体使用方法请输入：“help” 查看。')
        if helper.get('is_moviebox'):
            print('已开启票房查询功能，具体使用方法请输入：“help” 查看。')
        if helper.get('is_air_quality'):
            print('已开启空气质量查询功能，具体使用方法请输入：“help” 查看。')
    print('=' * 80)
    alarm = config.get('alarm_info')
    if not alarm or not alarm.get('is_alarm'):
        print('未开启每日提醒功能。')
    else:
        print('已开启定时发送提醒功能。')
        alarm_dict = alarm.get('alarm_dict')
        for value in alarm_dict.values():
            nickname_list = value.get('nickname_list')
            nns = '，'.join(nickname_list)
            hour = value.get('hour')
            minute = value.get('minute')
            alarm_time = '{hour:0>2d}:{minute:0>2d}'.format(hour=hour, minute=minute)
            jitter = value.get('alarm_jitter', 0)
            if jitter != 0:
                set_time = datetime.strptime(alarm_time, '%H:%M')
                jitter_time = timedelta(seconds=jitter)
                start_time = (set_time - jitter_time).strftime('%H:%M')
                end_time = (set_time + jitter_time).strftime('%H:%M')
                alarm_time = '{start_time}——{end_time} 期间'.format(start_time=start_time, end_time=end_time)
            print('定时：{alarm_time}，给：{nicknames}，发送提醒内容一次。'.format(alarm_time=alarm_time, nicknames=nns))
    print('=' * 80)