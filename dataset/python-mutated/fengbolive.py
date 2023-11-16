from Crypto.Cipher import AES
from urllib.parse import unquote
import base64
import json
import requests

class FengBo:

    def __init__(self, rid):
        if False:
            for i in range(10):
                print('nop')
        self.rid = rid

    def get_real_url(self):
        if False:
            print('Hello World!')
        with requests.Session() as s:
            res = s.get(f'https://external.fengbolive.com/cgi-bin/get_anchor_info_proxy.fcgi?anchorid={self.rid}')
            res = res.json()
        if res['ret'] == 1:
            info = res['info']
            info = unquote(info, 'utf-8')

            def pad(t):
                if False:
                    return 10
                return t + (16 - len(t) % 16) * b'\x00'
            key = iv = 'abcdefghqwertyui'.encode('utf8')
            cipher = AES.new(key, AES.MODE_CBC, iv)
            info = info.encode('utf8')
            info = pad(info)
            result = cipher.decrypt(base64.decodebytes(info)).rstrip(b'\x00')
            result = json.loads(result.decode('utf-8'))
            url = result['url']
            url = url.replace('hdl', 'hls')
            url = url.replace('.flv', '/playlist.m3u8')
            return url
        else:
            raise Exception('房间号错误')

def get_real_url(rid):
    if False:
        i = 10
        return i + 15
    try:
        fb = FengBo(rid)
        return fb.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('输入疯播直播房间号：\n')
    print(get_real_url(r))