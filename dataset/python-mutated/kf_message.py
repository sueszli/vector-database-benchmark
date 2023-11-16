from optionaldict import optionaldict
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatKFMessage(BaseWeChatAPI):
    """
    发送微信客服消息

    https://work.weixin.qq.com/api/doc/90000/90135/94677

    支持：
    * 文本消息
    * 图片消息
    * 语音消息
    * 视频消息
    * 文件消息
    * 图文链接
    * 小程序
    * 菜单消息
    * 地理位置
    """

    def send(self, user_id, open_kfid, msgid='', msg=None):
        if False:
            print('Hello World!')
        '\n        当微信客户处于“新接入待处理”或“由智能助手接待”状态下，可调用该接口给用户发送消息。\n        注意仅当微信客户在主动发送消息给客服后的48小时内，企业可发送消息给客户，最多可发送5条消息；若用户继续发送消息，企业可再次下发消息。\n        支持发送消息类型：文本、图片、语音、视频、文件、图文、小程序、菜单消息、地理位置。\n\n        :param user_id: 指定接收消息的客户UserID\n        :param open_kfid: 指定发送消息的客服帐号ID\n        :param msgid: 指定消息ID\n        :param tag_ids: 标签ID列表。\n        :param msg: 发送消息的 dict 对象\n        :type msg: dict | None\n        :return: 接口调用结果\n        '
        msg = msg or {}
        data = {'touser': user_id, 'open_kfid': open_kfid}
        if msgid:
            data['msgid'] = msgid
        data.update(msg)
        return self._post('kf/send_msg', data=data)

    def send_text(self, user_id, open_kfid, content, msgid=''):
        if False:
            for i in range(10):
                print('nop')
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'text', 'text': {'content': content}})

    def send_image(self, user_id, open_kfid, media_id, msgid=''):
        if False:
            print('Hello World!')
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'image', 'image': {'media_id': media_id}})

    def send_voice(self, user_id, open_kfid, media_id, msgid=''):
        if False:
            print('Hello World!')
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'voice', 'voice': {'media_id': media_id}})

    def send_video(self, user_id, open_kfid, media_id, msgid=''):
        if False:
            while True:
                i = 10
        video_data = optionaldict()
        video_data['media_id'] = media_id
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'video', 'video': dict(video_data)})

    def send_file(self, user_id, open_kfid, media_id, msgid=''):
        if False:
            i = 10
            return i + 15
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'file', 'file': {'media_id': media_id}})

    def send_articles_link(self, user_id, open_kfid, article, msgid=''):
        if False:
            for i in range(10):
                print('nop')
        articles_data = {'title': article['title'], 'desc': article['desc'], 'url': article['url'], 'thumb_media_id': article['thumb_media_id']}
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'news', 'link': {'link': articles_data}})

    def send_msgmenu(self, user_id, open_kfid, head_content, menu_list, tail_content, msgid=''):
        if False:
            print('Hello World!')
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'msgmenu', 'msgmenu': {'head_content': head_content, 'list': menu_list, 'tail_content': tail_content}})

    def send_location(self, user_id, open_kfid, name, address, latitude, longitude, msgid=''):
        if False:
            print('Hello World!')
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'location', 'msgmenu': {'name': name, 'address': address, 'latitude': latitude, 'longitude': longitude}})

    def send_miniprogram(self, user_id, open_kfid, appid, title, thumb_media_id, pagepath, msgid=''):
        if False:
            return 10
        return self.send(user_id, open_kfid, msgid, msg={'msgtype': 'miniprogram', 'msgmenu': {'appid': appid, 'title': title, 'thumb_media_id': thumb_media_id, 'pagepath': pagepath}})