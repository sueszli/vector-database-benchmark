import mw
from flask import request
import requests

class vip_api:
    api_url = 'https://wo.midoks.me/api/wp-json/vip'

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def loginApi(self):
        if False:
            return 10
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        password = mw.aesEncrypt(password)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        print('name:', str(username))
        print('pwd:', str(password))
        args = {'name': username, 'pass': password}
        data = requests.post(self.api_url + '/v1/login', data=args, headers=headers)
        print(data.text)
        return mw.returnJson(False, '测试中!')