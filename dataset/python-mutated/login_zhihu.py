import requests, time
import hmac, json
from bs4 import BeautifulSoup
from hashlib import sha1

def get_captcha(data, need_cap):
    if False:
        while True:
            i = 10
    ' 处理验证码 '
    if need_cap is False:
        return
    with open('captcha.gif', 'wb') as fb:
        fb.write(data)
    return input('captcha:')

def get_signature(grantType, clientId, source, timestamp):
    if False:
        return 10
    ' 处理签名 '
    hm = hmac.new(b'd1b964811afb40118a12068ff74a12f4', None, sha1)
    hm.update(str.encode(grantType))
    hm.update(str.encode(clientId))
    hm.update(str.encode(source))
    hm.update(str.encode(timestamp))
    return str(hm.hexdigest())

def login(username, password, oncaptcha, sessiona, headers):
    if False:
        while True:
            i = 10
    ' 处理登录 '
    resp1 = sessiona.get('https://www.zhihu.com/signin', headers=headers)
    resp2 = sessiona.get('https://www.zhihu.com/api/v3/oauth/captcha?lang=cn', headers=headers)
    need_cap = json.loads(resp2.text)['show_captcha']
    grantType = 'password'
    clientId = 'c3cef7c66a1843f8b3a9e6a1e3160e20'
    source = 'com.zhihu.web'
    timestamp = str(time.time() * 1000).split('.')[0]
    captcha_content = sessiona.get('https://www.zhihu.com/captcha.gif?r=%d&type=login' % (time.time() * 1000), headers=headers).content
    data = {'client_id': clientId, 'grant_type': grantType, 'timestamp': timestamp, 'source': source, 'signature': get_signature(grantType, clientId, source, timestamp), 'username': username, 'password': password, 'lang': 'cn', 'captcha': oncaptcha(captcha_content, need_cap), 'ref_source': 'other_', 'utm_source': ''}
    print('**2**: ' + str(data))
    print('-' * 50)
    resp = sessiona.post('https://www.zhihu.com/api/v3/oauth/sign_in', data, headers=headers).content
    print(BeautifulSoup(resp, 'html.parser'))
    print('-' * 50)
    return resp
if __name__ == '__main__':
    sessiona = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0', 'authorization': 'oauth c3cef7c66a1843f8b3a9e6a1e3160e20'}
    login('12345678@qq.com', '12345678', get_captcha, sessiona, headers)
    resp = sessiona.get('https://www.zhihu.com/inbox', headers=headers)
    print(BeautifulSoup(resp.content, 'html.parser'))