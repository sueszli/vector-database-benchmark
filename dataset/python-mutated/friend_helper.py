"""
Project: EverydayWechat-Github
Creator: DoubleThunder
Create time: 2019-07-12 23:07
Introduction: 处理好友消息内容
"""
import time
import random
import itchat
from everyday_wechat.utils import config
from everyday_wechat.utils.data_collection import get_bot_info
from everyday_wechat.utils.common import FILEHELPER
__all__ = ['handle_friend']

def handle_friend(msg):
    if False:
        i = 10
        return i + 15
    ' 处理好友信息 '
    try:
        if msg['FromUserName'] == config.get('wechat_uuid') and msg['ToUserName'] != FILEHELPER:
            return
        conf = config.get('auto_reply_info')
        if not conf.get('is_auto_reply'):
            return
        uuid = FILEHELPER if msg['ToUserName'] == FILEHELPER else msg['FromUserName']
        is_all = conf.get('is_auto_reply_all')
        auto_uuids = conf.get('auto_reply_black_uuids') if is_all else conf.get('auto_reply_white_uuids')
        if is_all and uuid in auto_uuids:
            return
        if not is_all and uuid not in auto_uuids:
            return
        receive_text = msg.text
        nick_name = FILEHELPER if uuid == FILEHELPER else msg.user.nickName
        print('\n{}发来信息：{}'.format(nick_name, receive_text))
        reply_text = get_bot_info(receive_text, uuid)
        if reply_text:
            time.sleep(random.randint(1, 2))
            prefix = conf.get('auto_reply_prefix', '')
            if prefix:
                reply_text = '{}{}'.format(prefix, reply_text)
            suffix = conf.get('auto_reply_suffix', '')
            if suffix:
                reply_text = '{}{}'.format(reply_text, suffix)
            itchat.send(reply_text, toUserName=uuid)
            print('回复{}：{}'.format(nick_name, reply_text))
        else:
            print('自动回复失败\n')
    except Exception as exception:
        print(str(exception))