"""
Project: EverydayWechat-Github
Creator: DoubleThunder
Create time: 2019-07-02 01:24
Introduction: 天行机器人 申请地址( https://www.tianapi.com/apiview/47 )
"""
import requests
from everyday_wechat.utils import config
from everyday_wechat.utils.common import md5_encode
__all__ = ['get_auto_reply', 'BOT_INDEX', 'BOT_NAME']
BOT_INDEX = 5
BOT_NAME = '天行机器人'

def get_tianapi_robot(text, userid):
    if False:
        return 10
    '\n    从天行机器人获取自动回复,接口地址:<https://www.tianapi.com/apiview/47>\n    :param text: 发出的消息\n    :param userid: 收到的内容\n    :return:\n    '
    try:
        info = config.get('auto_reply_info')['txapi_conf']
        app_key = info['app_key']
        if not app_key:
            print('天行机器人 app_key 为空，请求失败')
            return
        reply_name = info.get('reply_name', '')
        bot_name = info.get('bot_name', '')
        params = {'key': app_key, 'question': text, 'userid': md5_encode(userid), 'limit': 10, 'mode': 1, 'datatype': '0'}
        url = 'https://api.tianapi.com/txapi/robot/'
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            content_dict = resp.json()
            if content_dict['code'] == 200:
                if content_dict['datatype'] == 'text':
                    data_dict = content_dict['newslist']
                    reply_text = data_dict[0]['reply']
                    reply_text.replace('{robotname}', bot_name).replace('{appellation}', reply_name)
                    return reply_text
                else:
                    return '我不太懂你在说什么'
            else:
                print('天行机器人获取数据失败:{}'.format(content_dict['msg']))
        print('获取数据失败')
        return None
    except Exception as exception:
        print(str(exception))
get_auto_reply = get_tianapi_robot
if __name__ == '__main__':
    text = '我是你的谁'
    userid = '250'
    from_text = get_tianapi_robot(text, userid)
    print(from_text)