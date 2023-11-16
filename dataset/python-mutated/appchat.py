from optionaldict import optionaldict
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatAppChat(BaseWeChatAPI):
    """发送消息到群聊会话
    https://work.weixin.qq.com/api/doc#90000/90135/90244
    """

    def create(self, chat_id=None, name=None, owner=None, user_list=None):
        if False:
            while True:
                i = 10
        '\n        创建群聊会话\n\n        详情请参考\n        https://work.weixin.qq.com/api/doc#90000/90135/90245\n\n        限制说明：\n        只允许企业自建应用调用，且应用的可见范围必须是根部门；\n        群成员人数不可超过管理端配置的“群成员人数上限”，且最大不可超过500人；\n        每企业创建群数不可超过1000/天；\n\n        :param chat_id: 群聊的唯一标志，不能与已有的群重复；字符串类型，最长32个字符。只允许字符0-9及字母a-zA-Z。如果不填，系统会随机生成群id\n        :param name: 群聊名，最多50个utf8字符，超过将截断\n        :param owner: 指定群主的id。如果不指定，系统会随机从userlist中选一人作为群主\n        :param user_list: 会话成员列表，成员用userid来标识。至少2人，至多500人\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict(chatid=chat_id, name=name, owner=owner, userlist=user_list)
        return self._post('appchat/create', data=data)

    def get(self, chat_id):
        if False:
            return 10
        '\n        获取群聊会话\n\n        详情请参考\n        https://work.weixin.qq.com/api/doc#90000/90135/90247\n\n        :param chat_id: 群聊id\n        :return: 会话信息\n        '
        res = self._get('appchat/get', params={'chatid': chat_id})
        return res['chat_info']

    def update(self, chat_id, name=None, owner=None, add_user_list=None, del_user_list=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        修改群聊会话\n\n        详情请参考\n        https://work.weixin.qq.com/api/doc#90000/90135/90246\n\n        :param chat_id: 群聊id\n        :param name: 新的群聊名。若不需更新，请忽略此参数。最多50个utf8字符，超过将截断\n        :param owner: 新群主的id。若不需更新，请忽略此参数\n        :param add_user_list: 会话新增成员列表，成员用userid来标识\n        :param del_user_list: 会话退出成员列表，成员用userid来标识\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict(chatid=chat_id, name=name, owner=owner, add_user_list=add_user_list, del_user_list=del_user_list)
        return self._post('appchat/update', data=data)

    def send(self, chat_id, msg_type, **kwargs):
        if False:
            print('Hello World!')
        '\n        应用推送消息\n\n        详情请参考：https://work.weixin.qq.com/api/doc#90000/90135/90248\n        :param chat_id: 群聊id\n        :param msg_type: 消息类型，可以为text/image/voice/video/file/textcard/news/mpnews/markdown\n        :param kwargs: 具体消息类型的扩展参数\n        :return:\n        '
        data = {'chatid': chat_id, 'safe': kwargs.get('safe') or 0}
        data.update(self._build_msg_content(msg_type, **kwargs))
        return self._post('appchat/send', data=data)

    def send_text(self, chat_id, content, safe=0):
        if False:
            i = 10
            return i + 15
        '\n        发送文本消息\n\n        详情请参考：https://work.weixin.qq.com/api/doc#90000/90135/90248/文本消息/\n\n        :param chat_id: 群聊id\n        :param content: 消息内容\n        :param safe: 表示是否是保密消息，0表示否，1表示是，默认0\n        :return:\n        '
        return self.send(chat_id, 'text', safe=safe, content=content)

    def _build_msg_content(self, msgtype='text', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        构造消息内容\n\n        :param content: 消息内容，最长不超过2048个字节\n        :param msgtype: 消息类型，可以为text/image/voice/video/file/textcard/news/mpnews/markdown\n        :param kwargs: 具体消息类型的扩展参数\n        :return:\n        '
        data = {'msgtype': msgtype}
        if msgtype == 'text':
            data[msgtype] = {'content': kwargs.get('content')}
        elif msgtype == 'image' or msgtype == 'voice' or msgtype == 'file':
            data[msgtype] = {'media_id': kwargs.get('media_id')}
        elif msgtype == 'video':
            data[msgtype] = {'media_id': kwargs.get('media_id'), 'title': kwargs.get('title'), 'description': kwargs.get('description')}
        elif msgtype == 'textcard':
            data[msgtype] = {'title': kwargs.get('title'), 'description': kwargs.get('description'), 'url': kwargs.get('url'), 'btntxt': kwargs.get('btntxt')}
        elif msgtype == 'news':
            data[msgtype] = kwargs
        elif msgtype == 'mpnews':
            data[msgtype] = kwargs
        elif msgtype == 'markdown':
            data[msgtype] = kwargs
        else:
            raise TypeError(f'不能识别的msgtype: {msgtype}')
        return data