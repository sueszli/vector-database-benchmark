import requests
import time
import hashlib
import re

class HuoMao:

    def __init__(self, rid):
        if False:
            print('Hello World!')
        '\n        火猫直播已经倒闭了\n        Args:\n            rid: 房间号\n        '
        self.rid = rid

    @staticmethod
    def get_videoids(rid):
        if False:
            while True:
                i = 10
        room_url = f'https://www.huomao.com/mobile/mob_live/{rid}'
        response = requests.get(url=room_url).text
        try:
            videoids = re.findall('var stream = "([\\w\\W]+?)";', response)[0]
        except IndexError:
            videoids = 0
        return videoids

    @staticmethod
    def get_token(videoids):
        if False:
            i = 10
            return i + 15
        tt = str(int(time.time() * 1000))
        token = hashlib.md5(f'{videoids}huomaoh5room{tt}6FE26D855E1AEAE090E243EB1AF73685'.encode('utf-8')).hexdigest()
        return token

    def get_real_url(self):
        if False:
            return 10
        videoids = self.get_videoids(self.rid)
        if videoids:
            token = self.get_token(videoids)
            room_url = 'https://www.huomao.com/swf/live_data'
            post_data = {'cdns': 1, 'streamtype': 'live', 'VideoIDS': videoids, 'from': 'huomaoh5room', 'time': time, 'token': token}
            response = requests.post(url=room_url, data=post_data).json()
            roomStatus = response.get('roomStatus', 0)
            if roomStatus == '1':
                real_url_flv = response.get('streamList')[-1].get('list')[0].get('url')
                real_url_m3u8 = response.get('streamList')[-1].get('list_hls')[0].get('url')
                real_url = [real_url_flv, real_url_m3u8.replace('_480', '')]
            else:
                raise Exception('直播间未开播')
        else:
            raise Exception('直播间不存在')
        return real_url

def get_real_url(rid):
    if False:
        while True:
            i = 10
    try:
        hm = HuoMao(rid)
        return hm.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('请输入火猫直播房间号：\n')
    print(get_real_url(r))