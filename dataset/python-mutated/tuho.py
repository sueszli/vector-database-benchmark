import requests
import re

class TuHo:

    def __init__(self, rid):
        if False:
            for i in range(10):
                print('nop')
        self.rid = rid

    def get_real_url(self):
        if False:
            while True:
                i = 10
        with requests.Session() as s:
            res = s.get(f'https://www.tuho.tv/{self.rid}').text
        flv = re.search('videoPlayFlv":"(https[\\s\\S]+?flv)', res)
        if flv:
            status = re.search('isPlaying\\s:\\s(\\w+),', res).group(1)
            if status == 'true':
                real_url = flv.group(1).replace('\\', '')
                return real_url
            else:
                raise Exception('未开播')
        else:
            raise Exception('直播间不存在')

def get_real_url(rid):
    if False:
        i = 10
        return i + 15
    try:
        th = TuHo(rid)
        return th.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('输入星光直播房间号：\n')
    print(get_real_url(r))