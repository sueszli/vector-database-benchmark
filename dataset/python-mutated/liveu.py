import requests

class liveU:

    def __init__(self, rid):
        if False:
            print('Hello World!')
        self.rid = rid

    def get_real_url(self):
        if False:
            while True:
                i = 10
        with requests.Session() as s:
            url = f'https://mobile.liveu.me/appgw/v2/watchstartweb?sessionid=&vid={self.rid}'
            res = s.get(url).json()
            play_url = res['retinfo']['play_url'] if res['retval'] == 'ok' else '不存在或未开播'
            return play_url

def get_real_url(rid):
    if False:
        i = 10
        return i + 15
    try:
        url = liveU(rid)
        return url.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('输入liveU直播房间号：\n')
    print(get_real_url(r))