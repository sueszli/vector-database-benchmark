import json
import requests

class ChangYou:

    def __init__(self, rid):
        if False:
            for i in range(10):
                print('nop')
        self.rid = rid

    def get_real_url(self):
        if False:
            i = 10
            return i + 15
        headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1 '}
        url = 'http://cxg.changyou.com/landingpage/getstreamname.action?roomid={}'.format(self.rid)
        with requests.Session() as s:
            res = s.get(url, headers=headers).json()
        try:
            code = res['code']
            if code == 'error':
                return res['msg']
            else:
                stream = res['obj']['stream']
                url = 'http://pull.wscdn.cxg.changyou.com/show/{}.flv'.format(stream)
                return url
        except json.decoder.JSONDecodeError:
            return '输入错误'

def get_real_url(rid):
    if False:
        while True:
            i = 10
    try:
        cxg = ChangYou(rid)
        return cxg.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('请输入畅秀阁roomid：\n')
    print(get_real_url(r))