import hashlib
import time
import datetime
from operator import itemgetter
from urllib.parse import quote
from optionaldict import optionaldict
from wechatpy.utils import to_binary
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatCustomService(BaseWeChatAPI):
    API_BASE_URL = 'https://api.weixin.qq.com/customservice/'

    def add_account(self, account, nickname, password):
        if False:
            i = 10
            return i + 15
        '\n        添加客服账号\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Service_Center_messages.html#%E6%B7%BB%E5%8A%A0%E5%AE%A2%E6%9C%8D%E5%B8%90%E5%8F%B7\n\n        :param account: 完整客服账号，格式为：账号前缀@公众号微信号\n        :param nickname: 客服昵称，最长6个汉字或12个英文字符\n        :param password: 客服账号登录密码\n        :return: 返回的 JSON 数据包\n        '
        password = to_binary(password)
        password = hashlib.md5(password).hexdigest()
        return self._post('kfaccount/add', data={'kf_account': account, 'nickname': nickname, 'password': password})

    def update_account(self, account, nickname, password):
        if False:
            for i in range(10):
                print('nop')
        '\n        修改客服账号\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Service_Center_messages.html#%E4%BF%AE%E6%94%B9%E5%AE%A2%E6%9C%8D%E5%B8%90%E5%8F%B7\n\n        :param account: 完整客服账号，格式为：账号前缀@公众号微信号\n        :param nickname: 客服昵称，最长6个汉字或12个英文字符\n        :param password: 客服账号登录密码\n        :return: 返回的 JSON 数据包\n        '
        password = to_binary(password)
        password = hashlib.md5(password).hexdigest()
        return self._post('kfaccount/update', data={'kf_account': account, 'nickname': nickname, 'password': password})

    def delete_account(self, account):
        if False:
            while True:
                i = 10
        '\n        删除客服账号\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Service_Center_messages.html#%E5%88%A0%E9%99%A4%E5%AE%A2%E6%9C%8D%E5%B8%90%E5%8F%B7\n\n        :param account: 完整客服账号，格式为：账号前缀@公众号微信号\n        :return: 返回的 JSON 数据包\n        '
        params_data = [f'access_token={quote(self.access_token)}', f"kf_account={quote(to_binary(account), safe=b'/@')}"]
        params = '&'.join(params_data)
        return self._get('kfaccount/del', params=params)

    def get_accounts(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取所有客服账号\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Service_Center_messages.html#%E8%8E%B7%E5%8F%96%E6%89%80%E6%9C%89%E5%AE%A2%E6%9C%8D%E8%B4%A6%E5%8F%B7\n\n        :return: 客服账号列表\n        '
        res = self._get('getkflist', result_processor=itemgetter('kf_list'))
        return res

    def upload_headimg(self, account, media_file):
        if False:
            print('Hello World!')
        '\n        设置客服帐号的头像\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Service_Center_messages.html#%E8%AE%BE%E7%BD%AE%E5%AE%A2%E6%9C%8D%E5%B8%90%E5%8F%B7%E7%9A%84%E5%A4%B4%E5%83%8F\n\n        :param account: 完整客服帐号，格式为：帐号前缀@公众号微信号\n        :param media_file: 要上传的头像文件，一个 File-Object\n        :return: 返回的 JSON 数据包\n        '
        return self._post('kfaccount/uploadheadimg', params={'kf_account': account}, files={'media': media_file})

    def get_online_accounts(self):
        if False:
            return 10
        '\n        获取在线客服接待信息\n        详情请参考\n        http://mp.weixin.qq.com/wiki/9/6fff6f191ef92c126b043ada035cc935.html\n\n        :return: 客服接待信息列表\n        '
        res = self._get('getonlinekflist', result_processor=itemgetter('kf_online_list'))
        return res

    def create_session(self, openid, account, text=None):
        if False:
            print('Hello World!')
        '\n        多客服创建会话\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Customer_Service/Session_control.html\n\n        :param openid: 客户 openid\n        :param account: 完整客服帐号，格式为：帐号前缀@公众号微信号\n        :param text: 附加信息，可选\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict(openid=openid, kf_account=account, text=text)
        return self._post('kfsession/create', data=data)

    def close_session(self, openid, account, text=None):
        if False:
            while True:
                i = 10
        '\n        多客服关闭会话\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Customer_Service/Session_control.html\n\n        :param openid: 客户 openid\n        :param account: 完整客服帐号，格式为：帐号前缀@公众号微信号\n        :param text: 附加信息，可选\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict(openid=openid, kf_account=account, text=text)
        return self._post('kfsession/close', data=data)

    def get_session(self, openid):
        if False:
            i = 10
            return i + 15
        '\n        获取客户的会话状态，如果不存在，则 kf_account 为空\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Customer_Service/Session_control.html\n\n        :param openid: 粉丝的 openid\n        :return: 返回的 JSON 数据包\n        '
        return self._get('kfsession/getsession', params={'openid': openid})

    def get_session_list(self, account):
        if False:
            i = 10
            return i + 15
        '\n        获取客服的会话列表\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Customer_Service/Session_control.html\n\n        :param account: 完整客服帐号，格式为：帐号前缀@公众号微信号\n        :return: 客服的会话列表\n        '
        res = self._get('kfsession/getsessionlist', params={'kf_account': account}, result_processor=itemgetter('sessionlist'))
        return res

    def get_wait_case(self):
        if False:
            i = 10
            return i + 15
        '\n        获取未接入会话列表\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Customer_Service/Session_control.html\n\n        :return: 返回的 JSON 数据包\n        '
        return self._get('kfsession/getwaitcase')

    def get_records(self, start_time, end_time, msgid=1, number=10000):
        if False:
            print('Hello World!')
        '\n        获取客服聊天记录\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/Customer_Service/Obtain_chat_transcript.html\n\n        :param start_time: 查询开始时间，UNIX 时间戳\n        :param end_time: 查询结束时间，UNIX 时间戳，每次查询不能跨日查询\n        :param msgid: 消息id顺序从小到大，从1开始\n        :param number: 每次获取条数，最多10000条\n\n        :return: 返回的 JSON 数据包\n        '
        if isinstance(start_time, datetime.datetime):
            start_time = time.mktime(start_time.timetuple())
        if isinstance(end_time, datetime.datetime):
            end_time = time.mktime(end_time.timetuple())
        record_data = {'starttime': int(start_time), 'endtime': int(end_time), 'msgid': msgid, 'number': number}
        res = self._post('msgrecord/getmsglist', data=record_data)
        return res