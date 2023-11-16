"""
Project: HelloWorldPython
Creator: DoubleThunder
Create time: 2019-07-02 02:46
Introduction: 海知智能 <https://ruyi.ai/> 功能很强大，不仅仅用于聊天。需申请 key，免费
"""
import requests
from everyday_wechat.utils import config
from everyday_wechat.utils.common import md5_encode
__all__ = ['get_auto_reply', 'BOT_INDEX', 'BOT_NAME']
BOT_INDEX = 6
BOT_NAME = '海知智能机器人'
URL = 'http://api.ruyi.ai/v1/message'

def get_ruyiai_bot(text, userId):
    if False:
        while True:
            i = 10
    '\n    海知智能 文档说明：<http://docs.ruyi.ai/502931>\n    :param text: str 需要发送的话\n    :param userId: str 用户标识\n    :return: str 机器人回复\n    '
    try:
        info = config.get('auto_reply_info')['ruyi_conf']
        app_key = info['app_key']
        if not app_key:
            print('海知智能 api_key 为空，请求失败')
            return
        params = {'q': text, 'user_id': md5_encode(userId), 'app_key': app_key}
        headers = {'Content-Type': 'application/json'}
        resp = requests.get(URL, headers=headers, params=params)
        if resp.status_code == 200:
            content_dict = resp.json()
            if content_dict['code'] in (0, 200):
                outputs = content_dict['result']['intents'][0]['outputs']
                reply_text = outputs[0]['property']['text']
                return reply_text
            else:
                print('海知智能 获取数据失败:{}'.format(content_dict['msg']))
                return
        print('海知智能 获取数据失败')
        return None
    except Exception as exception:
        print(str(exception))
get_auto_reply = get_ruyiai_bot
if __name__ == '__main__':
    pass