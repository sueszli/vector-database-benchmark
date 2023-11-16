import hashlib
import re
import time
import execjs
import requests

class DouYu:
    """
    可用来替换返回链接中的主机部分
    两个阿里的CDN：
    dyscdnali1.douyucdn.cn
    dyscdnali3.douyucdn.cn
    墙外不用带尾巴的akm cdn：
    hls3-akm.douyucdn.cn
    hlsa-akm.douyucdn.cn
    hls1a-akm.douyucdn.cn
    """

    def __init__(self, rid):
        if False:
            while True:
                i = 10
        '\n        房间号通常为1~8位纯数字，浏览器地址栏中看到的房间号不一定是真实rid.\n        Args:\n            rid:\n        '
        self.did = '10000000000000000000000000001501'
        self.t10 = str(int(time.time()))
        self.t13 = str(int(time.time() * 1000))
        self.s = requests.Session()
        self.res = self.s.get('https://m.douyu.com/' + str(rid)).text
        result = re.search('rid":(\\d{1,8}),"vipId', self.res)
        if result:
            self.rid = result.group(1)
        else:
            raise Exception('房间号错误')

    @staticmethod
    def md5(data):
        if False:
            i = 10
            return i + 15
        return hashlib.md5(data.encode('utf-8')).hexdigest()

    def get_pre(self):
        if False:
            return 10
        url = 'https://playweb.douyucdn.cn/lapi/live/hlsH5Preview/' + self.rid
        data = {'rid': self.rid, 'did': self.did}
        auth = DouYu.md5(self.rid + self.t13)
        headers = {'rid': self.rid, 'time': self.t13, 'auth': auth}
        res = self.s.post(url, headers=headers, data=data).json()
        error = res['error']
        data = res['data']
        key = ''
        if data:
            rtmp_live = data['rtmp_live']
            key = re.search('(\\d{1,8}[0-9a-zA-Z]+)_?\\d{0,4}(/playlist|.m3u8)', rtmp_live).group(1)
        return (error, key)

    def get_js(self):
        if False:
            print('Hello World!')
        result = re.search('(function ub98484234.*)\\s(var.*)', self.res).group()
        func_ub9 = re.sub('eval.*;}', 'strc;}', result)
        js = execjs.compile(func_ub9)
        res = js.call('ub98484234')
        v = re.search('v=(\\d+)', res).group(1)
        rb = DouYu.md5(self.rid + self.did + self.t10 + v)
        func_sign = re.sub('return rt;}\\);?', 'return rt;}', res)
        func_sign = func_sign.replace('(function (', 'function sign(')
        func_sign = func_sign.replace('CryptoJS.MD5(cb).toString()', '"' + rb + '"')
        js = execjs.compile(func_sign)
        params = js.call('sign', self.rid, self.did, self.t10)
        params += '&ver=219032101&rid={}&rate=-1'.format(self.rid)
        url = 'https://m.douyu.com/api/room/ratestream'
        res = self.s.post(url, params=params).text
        key = re.search('(\\d{1,8}[0-9a-zA-Z]+)_?\\d{0,4}(.m3u8|/playlist)', res).group(1)
        return key

    def get_pc_js(self, cdn='ws-h5', rate=0):
        if False:
            while True:
                i = 10
        '\n        通过PC网页端的接口获取完整直播源。\n        :param cdn: 主线路ws-h5、备用线路tct-h5\n        :param rate: 1流畅；2高清；3超清；4蓝光4M；0蓝光8M或10M\n        :return: JSON格式\n        '
        res = self.s.get('https://www.douyu.com/' + str(self.rid)).text
        result = re.search('(vdwdae325w_64we[\\s\\S]*function ub98484234[\\s\\S]*?)function', res).group(1)
        func_ub9 = re.sub('eval.*?;}', 'strc;}', result)
        js = execjs.compile(func_ub9)
        res = js.call('ub98484234')
        v = re.search('v=(\\d+)', res).group(1)
        rb = DouYu.md5(self.rid + self.did + self.t10 + v)
        func_sign = re.sub('return rt;}\\);?', 'return rt;}', res)
        func_sign = func_sign.replace('(function (', 'function sign(')
        func_sign = func_sign.replace('CryptoJS.MD5(cb).toString()', '"' + rb + '"')
        js = execjs.compile(func_sign)
        params = js.call('sign', self.rid, self.did, self.t10)
        params += '&cdn={}&rate={}'.format(cdn, rate)
        url = 'https://www.douyu.com/lapi/live/getH5Play/{}'.format(self.rid)
        res = self.s.post(url, params=params).json()
        return res

    def get_real_url(self):
        if False:
            i = 10
            return i + 15
        (error, key) = self.get_pre()
        if error == 0:
            pass
        elif error == 102:
            raise Exception('房间不存在')
        elif error == 104:
            raise Exception('房间未开播')
        else:
            key = self.get_js()
        real_url = {}
        real_url['flv'] = 'http://vplay1a.douyucdn.cn/live/{}.flv?uuid='.format(key)
        real_url['x-p2p'] = 'http://tx2play1.douyucdn.cn/live/{}.xs?uuid='.format(key)
        return real_url
if __name__ == '__main__':
    r = input('输入斗鱼直播间号：\n')
    s = DouYu(r)
    print(s.get_real_url())