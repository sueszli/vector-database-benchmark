"""
python_wechat.py by xianhu
主要包括如下功能：
(1) 自动提醒群红包
(2) 自动监测被撤回消息
(3) 群关键字提醒，群被@提醒
"""
import time
import itchat
import logging
from itchat.content import *
my = itchat.new_instance()
my.auto_login(hotReload=False, enableCmdQR=2)
my.global_keys = ['创业', '人工智能', '企业服务']
my.to_user_name = 'filehelper'
my.update_time = time.time()
my.msg_store = {}
my.friends = {}
my.groups = {}

def update_my_infos():
    if False:
        i = 10
        return i + 15
    '\n    更新信息\n    '
    my.friends = {user['UserName']: user for user in my.get_friends(update=True)}
    my.groups = {group['UserName']: group for group in my.get_chatrooms(update=True)}
    return
update_my_infos()

class Message(object):
    """
    消息类
    """

    def __init__(self, msg):
        if False:
            print('Hello World!')
        '\n        构造函数：提取消息内容\n        消息来源分类：\n        （1）来自好友的消息\n        （2）来自群的消息\n        提取消息内容，消息类型分类：\n        （1）文字（2）图片（3）语音（4）视频（5）地址（6）名片（7）提醒（8）分享（9）附件\n        '
        if time.time() - my.update_time > 600:
            update_my_infos()
            my.update_time = time.time()
        self.msg_id = msg['MsgId']
        self.from_user_name = msg['FromUserName']
        self.msg_type = msg['MsgType']
        self.msg_content = msg['Content']
        self.msg_time = msg['CreateTime']
        self.msg_file = msg['FileName']
        self.msg_file_length = msg['FileSize']
        self.msg_voice_length = msg['VoiceLength']
        self.msg_play_length = msg['PlayLength']
        self.msg_url = msg['Url']
        self.user_user_name = msg['User'].get('UserName', '')
        self.user_nick_name = msg['User'].get('NickName', '')
        self.user_remark_name = msg['User'].get('RemarkName', '')
        self.wind_name = self.user_remark_name if self.user_remark_name else self.user_nick_name if self.user_nick_name else my.friends[self.user_user_name]['NickName'] if self.user_user_name in my.friends else my.groups[self.user_user_name]['NickName'] if self.user_user_name in my.groups else '未知窗口'
        self.actual_user_name = msg.get('ActualUserName', '')
        self.actual_nick_name = msg.get('ActualNickName', '')
        self.actual_remark_name = self.actual_nick_name if self.actual_user_name not in my.friends or not my.friends[self.actual_user_name]['RemarkName'] else my.friends[self.actual_user_name]['RemarkName']
        self.is_at = msg.get('IsAt', None)
        self.we_type = msg['Type']
        self.we_text = msg['Text']
        logging.warning('wind_name=%s, send_name=%s, we_type=%s, we_text=%s', self.wind_name, self.actual_remark_name, self.we_type, self.we_text)
        return

def process_message_group(msg):
    if False:
        return 10
    '\n    处理群消息\n    '
    if msg.we_type == 'Note' and msg.we_text.find('收到红包，请在手机上查看') >= 0:
        my.send('【%s】中有人发红包啦，快抢！' % msg.wind_name, toUserName=my.to_user_name)
    for key in my.global_keys:
        if msg.we_type == 'Text' and msg.we_text.find(key) >= 0:
            my.send('【%s】中【%s】提及了关键字：%s' % (msg.wind_name, msg.actual_remark_name, key), toUserName=my.to_user_name)
            my.send(msg.we_text, toUserName=my.to_user_name)
            break
    if msg.we_type == 'Text' and msg.is_at:
        my.send('【%s】中【%s】@了你' % (msg.wind_name, msg.actual_remark_name), toUserName=my.to_user_name)
        my.send(msg.we_text, toUserName=my.to_user_name)
    return

def process_message_revoke(msg):
    if False:
        while True:
            i = 10
    '\n    处理撤回消息\n    '
    my.msg_store[msg.msg_id] = msg
    for _id in [_id for _id in my.msg_store if time.time() - my.msg_store[_id].msg_time > 120]:
        my.msg_store.pop(_id)
    if msg.we_type in ['Picture', 'Recording']:
        try:
            msg.we_text('.Cache/' + msg.msg_file)
            logging.warning('process_message_revoke: download %s to .Cache/', msg.msg_file)
        except Exception as excep:
            logging.error('process_message_revoke: download %s to .Cache/ error: %s', msg.msg_file, excep)
    if msg.we_type == 'Note' and msg.we_text.find('撤回了一条消息') >= 0:
        old_msg = my.msg_store.get(msg.msg_content[msg.msg_content.find('<msgid>') + 7:msg.msg_content.find('</msgid>')])
        if not old_msg:
            logging.warning('process_message_revoke: no message id in my.msg_store')
            return
        if old_msg.from_user_name.startswith('@@'):
            my.send('【%s】中【%s】撤回了自己发送的消息:\nType: %s\n%s' % (old_msg.wind_name, old_msg.actual_remark_name, old_msg.we_type, old_msg.msg_file), toUserName=my.to_user_name)
        else:
            my.send('【%s】撤回了自己发送的消息:\nType: %s\n%s' % (old_msg.wind_name, old_msg.we_type, old_msg.msg_file), toUserName=my.to_user_name)
        if old_msg.we_type in ['Text', 'Card']:
            my.send(str(old_msg.we_text), toUserName=my.to_user_name)
        elif old_msg.we_type == 'Sharing':
            my.send(old_msg.we_text + '\n' + old_msg.msg_url, toUserName=my.to_user_name)
        elif old_msg.we_type == 'Picture':
            my.send_image('.Cache/' + old_msg.msg_file, toUserName=my.to_user_name)
        elif old_msg.we_type == 'Recording':
            my.send_file('.Cache/' + old_msg.msg_file, toUserName=my.to_user_name)
    return

@my.msg_register([TEXT, PICTURE, RECORDING, VIDEO, MAP, CARD, NOTE, SHARING, ATTACHMENT], isFriendChat=True, isGroupChat=True)
def text_reply(msg):
    if False:
        while True:
            i = 10
    '\n    消息自动接收, 接受全部的消息（自己发送的消息除外）\n    '
    if msg['FromUserName'] == my.loginInfo['User']['UserName']:
        return
    msg = Message(msg)
    if msg.we_type not in ['Text', 'Picture', 'Recording', 'Card', 'Note', 'Sharing']:
        logging.warning("process_message_group: message type isn't included, ignored")
        return
    if msg.from_user_name.startswith('@@'):
        process_message_group(msg)
    process_message_revoke(msg)
    return
my.run(debug=False)
"\n好友消息：\n{\n    'MsgId': '5254859004542036569',\n    'FromUserName': '@f3b7fdc54717ea8dc22cb3edef59688e82ef34874e3236801537b94f6cd73e1e',\n    'ToUserName': '@e79dde912b8f817514c01f399ca9ba12',\n    'MsgType': 1,\n    'Content': '[微笑]己改',\n    'Status': 3,\n    'ImgStatus': 1,\n    'CreateTime': 1498448860,\n    'VoiceLength': 0,\n    'PlayLength': 0,\n    'FileName': '',\n    'FileSize': '',\n    'MediaId': '',\n    'Url': '',\n    'AppMsgType': 0,\n    'StatusNotifyCode': 0,\n    'StatusNotifyUserName': '',\n    'HasProductId': 0,\n    'Ticket': '',\n    'ImgHeight': 0,\n    'ImgWidth': 0,\n    'SubMsgType': 0,\n    'NewMsgId': 5254859004542036569,\n    'OriContent': '',\n    'User': <User: {\n        'MemberList': <ContactList: []>,\n        'Uin': 0,\n        'UserName': '@f3b7fdc54717ea8dc22cb3edef59688e82ef34874e3236801537b94f6cd73e1e',\n        'NickName': '付贵吉祥',\n        'HeadImgUrl': '/cgi-bin/mmwebwx-bin/webwxgeticon?seq=688475226&username=@f3b7fdc54717ea8dc22cb3edef59688e82ef34874e3236801537b94f6cd73e1e&skey=@',\n        'ContactFlag': 3,\n        'MemberCount': 0,\n        'RemarkName': '付贵吉祥@中建5号楼',\n        'HideInputBarFlag': 0,\n        'Sex': 1,\n        'Signature': '漫漫人生路...',\n        'VerifyFlag': 0,\n        'OwnerUin': 0,\n        'PYInitial': 'FGJX',\n        'PYQuanPin': 'fuguijixiang',\n        'RemarkPYInitial': 'FGJXZJ5HL',\n        'RemarkPYQuanPin': 'fuguijixiangzhongjian5haolou',\n        'StarFriend': 0,\n        'AppAccountFlag': 0,\n        'Statues': 0,\n        'AttrStatus': 135205,\n        'Province': '山东',\n        'City': '',\n        'Alias': '',\n        'SnsFlag': 17,\n        'UniFriend': 0,\n        'DisplayName': '',\n        'ChatRoomId': 0,\n        'KeyWord': '',\n        'EncryChatRoomId': '',\n        'IsOwner': 0\n    }>,\n    'Type': 'Text',\n    'Text': '[微笑]己改'\n}\n"
'\n群消息：\n{\n    \'MsgId\': \'7844877618948840992\',\n    \'FromUserName\': \'@@8dc5df044444d1fb8e3972e755b47adf9d07f5a032cae90a4d822b74ee1e4880\',\n    \'ToUserName\': \'@e79dde912b8f817514c01f399ca9ba12\',\n    \'MsgType\': 1,\n    \'Content\': \'就是那个，那个协议我们手上有吗\',\n    \'Status\': 3,\n    \'ImgStatus\': 1,\n    \'CreateTime\': 1498448972,\n    \'VoiceLength\': 0,\n    \'PlayLength\': 0,\n    \'FileName\': \'\',\n    \'FileSize\': \'\',\n    \'MediaId\': \'\',\n    \'Url\': \'\',\n    \'AppMsgType\': 0,\n    \'StatusNotifyCode\': 0,\n    \'StatusNotifyUserName\': \'\',\n    \'HasProductId\': 0,\n    \'Ticket\': \'\',\n    \'ImgHeight\': 0,\n    \'ImgWidth\': 0,\n    \'SubMsgType\': 0,\n    \'NewMsgId\': 7844877618948840992,\n    \'OriContent\': \'\',\n    \'ActualNickName\': \'5-1-1003\',\n    \'IsAt\': False,\n    \'ActualUserName\': \'@a0922f18795e4c3b6d7d09c492ace233\',\n    \'User\': <Chatroom: {\n        \'MemberList\': <ContactList: [\n            <ChatroomMember: {\n                \'MemberList\': <ContactList: []>,\n                \'Uin\': 0,\n                \'UserName\': \'@e79dde912b8f817514c01f399ca9ba12\',\n                \'NickName\': \'齐现虎\',\n                \'AttrStatus\': 2147600869,\n                \'PYInitial\': \'\',\n                \'PYQuanPin\': \'\',\n                \'RemarkPYInitial\': \'\',\n                \'RemarkPYQuanPin\': \'\',\n                \'MemberStatus\': 0,\n                \'DisplayName\': \'5-1-1601\',\n                \'KeyWord\': \'qix\'\n            }>,\n            <ChatroomMember: {\n                \'MemberList\': <ContactList: []>,\n                \'Uin\': 0,\n                \'UserName\': \'@a9620e3d4b82eab2521ccdbb985afc37\',\n                \'NickName\': \'A高佳祥15069179911\',\n                \'AttrStatus\': 102503,\n                \'PYInitial\': \'\',\n                \'PYQuanPin\': \'\',\n                \'RemarkPYInitial\': \'\',\n                \'RemarkPYQuanPin\': \'\',\n                \'MemberStatus\': 0,\n                \'DisplayName\': \'5-2-220315069179911\',\n                \'KeyWord\': \'gao\'\n            }>,\n            .......\n        ]>,\n        \'Uin\': 0,\n        \'UserName\': \'@@8dc5df044444d1fb8e3972e755b47adf9d07f5a032cae90a4d822b74ee1e4880\',\n        \'NickName\': \'中建锦绣澜庭二期5#楼\',\n        \'HeadImgUrl\': \'/cgi-bin/mmwebwx-bin/webwxgetheadimg?seq=0&username=@@8dc5df044444d1fb8e3972e755b47adf9d07f5a032cae90a4d822b74ee1e4880&skey=@\',\n        \'ContactFlag\': 3,\n        \'MemberCount\': 106,\n        \'RemarkName\': \'\',\n        \'HideInputBarFlag\': 0,\n        \'Sex\': 0,\n        \'Signature\': \'\',\n        \'VerifyFlag\': 0,\n        \'OwnerUin\': 0,\n        \'PYInitial\': \'ZJJXLTEJ5L\',\n        \'PYQuanPin\': \'zhongjianjinxiulantingerji5lou\',\n        \'RemarkPYInitial\': \'\',\n        \'RemarkPYQuanPin\': \'\',\n        \'StarFriend\': 0,\n        \'AppAccountFlag\': 0,\n        \'Statues\': 0,\n        \'AttrStatus\': 0,\n        \'Province\': \'\',\n        \'City\': \'\',\n        \'Alias\': \'\',\n        \'SnsFlag\': 0,\n        \'UniFriend\': 0,\n        \'DisplayName\': \'\',\n        \'ChatRoomId\': 0,\n        \'KeyWord\': \'\',\n        \'EncryChatRoomId\': \'@d1e510bc8cbd192468e9c85c6f5a9d81\',\n        \'IsOwner\': 1,\n        \'IsAdmin\': None,\n        \'Self\': <ChatroomMember: {\n            \'MemberList\': <ContactList: []>,\n            \'Uin\': 0,\n            \'UserName\': \'@e79dde912b8f817514c01f399ca9ba12\',\n            \'NickName\': \'齐现虎\',\n            \'AttrStatus\': 2147600869,\n            \'PYInitial\': \'\',\n            \'PYQuanPin\': \'\',\n            \'RemarkPYInitial\': \'\',\n            \'RemarkPYQuanPin\': \'\',\n            \'MemberStatus\': 0,\n            \'DisplayName\': \'5-1-1601\',\n            \'KeyWord\': \'qix\'\n        }>,\n        \'HeadImgUpdateFlag\': 1,\n        \'ContactType\': 0,\n        \'ChatRoomOwner\': \'@e79dde912b8f817514c01f399ca9ba12\'\n    }>,\n    \'Type\': \'Text\',\n    \'Text\': \'就是那个，那个协议我们手上有吗\'\n}\n\n警示消息：好友类\n{\n    \'MsgId\': \'1529895072288746571\',\n    \'FromUserName\': \'@4076708be2e09ef83f249f168553d0dd55b4f734aee7d276e92ddbe98625476a\',\n    \'ToUserName\': \'@f97583d8ffbaee6189854116897c677f\',\n    \'MsgType\': 10000,\n    \'Content\': \'你已添加了呼啸而过的小青春，现在可以开始聊天了。\',\n    \'Status\': 4,\n    \'ImgStatus\': 1,\n    \'CreateTime\': 1498533407,\n    \'VoiceLength\': 0,\n    \'PlayLength\': 0,\n    \'FileName\': \'\',\n    \'FileSize\': \'\',\n    \'MediaId\': \'\',\n    \'Url\': \'\',\n    \'AppMsgType\': 0,\n    \'StatusNotifyCode\': 0,\n    \'StatusNotifyUserName\': \'\',\n    \'HasProductId\': 0,\n    \'Ticket\': \'\',\n    \'ImgHeight\': 0,\n    \'ImgWidth\': 0,\n    \'SubMsgType\': 0,\n    \'NewMsgId\': 1529895072288746571,\n    \'OriContent\': \'\',\n    \'User\': <User: {\n        \'userName\': \'@4076708be2e09ef83f249f168553d0dd55b4f734aee7d276e92ddbe98625476a\',\n        \'MemberList\': <ContactList: []>\n    }>,\n    \'Type\': \'Note\',\n    \'Text\': \'你已添加了呼啸而过的小青春，现在可以开始聊天了。\'\n}\n\n警示消息：群类\n{\n    \'MsgId\': \'1049646282086057263\',\n    \'FromUserName\': \'@@300f57b68ca7ef593ae3221eef7dba5377466c86122aaa15a8ffc1031310e210\',\n    \'ToUserName\': \'@006f63e8086ab07fcbe3771dc824c4a6\',\n    \'MsgType\': 10000,\n    \'Content\': \'你邀请"大姐"加入了群聊\',\n    \'Status\': 3,\n    \'ImgStatus\': 1,\n    \'CreateTime\': 1498533901,\n    \'VoiceLength\': 0,\n    \'PlayLength\': 0,\n    \'FileName\': \'\',\n    \'FileSize\': \'\',\n    \'MediaId\': \'\',\n    \'Url\': \'\',\n    \'AppMsgType\': 0,\n    \'StatusNotifyCode\': 0,\n    \'StatusNotifyUserName\': \'\',\n    \'HasProductId\': 0,\n    \'Ticket\': \'\',\n    \'ImgHeight\': 0,\n    \'ImgWidth\': 0,\n    \'SubMsgType\': 0,\n    \'NewMsgId\': 1049646282086057263,\n    \'OriContent\': \'\',\n    \'ActualUserName\': \'@006f63e8086ab07fcbe3771dc824c4a6\',\n    \'ActualNickName\': \'某某某\',\n    \'IsAt\': False,\n    \'User\': <Chatroom: {\n        \'UserName\': \'@@300f57b68ca7ef593ae3221eef7dba5377466c86122aaa15a8ffc1031310e210\',\n        \'MemberList\': <ContactList: []>\n    }>,\n    \'Type\': \'Note\',\n    \'Text\': \'你邀请"大姐"加入了群聊\'\n}\n'