import requests
import re

class YuanBoBo:

    def __init__(self, rid):
        if False:
            print('Hello World!')
        self.rid = rid

    def get_real_url(self):
        if False:
            print('Hello World!')
        with requests.Session() as s:
            res = s.get(f'https://zhibo.yuanbobo.com/{self.rid}').text
        stream_id = re.search("stream_id:\\s+'(\\d+)'", res)
        if stream_id:
            status = re.search("status:\\s+'(\\d)'", res).group(1)
            if status == '1':
                real_url = f'https://tliveplay.yuanbobo.com/live/{stream_id.group(1)}.m3u8'
                return real_url
            else:
                raise Exception('未开播')
        else:
            raise Exception('直播间不存在')

def get_real_url(rid):
    if False:
        for i in range(10):
            print('nop')
    try:
        th = YuanBoBo(rid)
        return th.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False
if __name__ == '__main__':
    r = input('输入热猫直播房间号：\n')
    print(get_real_url(r))