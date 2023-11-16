"""
http://api.qingyunke.com/
青云客智能聊天机器人API
可直接使用
"""
import requests
__all__ = ['get_auto_reply', 'BOT_INDEX', 'BOT_NAME']
BOT_INDEX = 3
BOT_NAME = '青云客机器人'
URL = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg={}'

def get_qingyunke(text, userid=''):
    if False:
        print('Hello World!')
    '\n    青云客智能聊天机器人API http://api.qingyunke.com/\n    :param text: str 聊天\n    :param userid: str 无用\n    :return: str\n    '
    try:
        resp = requests.get(URL.format(text))
        if resp.status_code == 200:
            re_data = resp.json()
            if re_data['result'] == 0:
                return_text = re_data['content']
                return return_text
            error_text = re_data['content']
            print('青云客机器人错误信息：{}'.format(error_text))
        print('青云客机器人获取失败')
    except Exception as exception:
        print(str(exception))
        print('青云客机器人获取失败')
get_auto_reply = get_qingyunke
if __name__ == '__main__':
    text = '微博加个关注呗'
    userid = '250'
    rt = get_qingyunke(text, userid)
    print('回复：', rt)
    pass