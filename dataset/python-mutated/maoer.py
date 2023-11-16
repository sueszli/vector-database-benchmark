import json
import requests

class MAOER:

    def __init__(self, rid):
        if False:
            while True:
                i = 10
        self.rid = rid

    def get_real_url(self):
        if False:
            while True:
                i = 10
        headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1 '}
        url = 'https://fm.missevan.com/api/v2/live/{}'.format(self.rid)
        with requests.Session() as s:
            res = s.get(url, headers=headers).json()
        try:
            code = res['code']
            if code != 0:
                return res['info']
            else:
                channel = res['info']['room']['channel']
                return channel
        except json.decoder.JSONDecodeError:
            return '输入错误'

def get_real_url(rid):
    if False:
        i = 10
        return i + 15
    try:
        mr = MAOER(rid)
        return mr.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('请输入猫耳直播房间号：\n')
    print(get_real_url(r))