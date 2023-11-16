import requests

class HoneyHook(object):

    def __init__(self, ip_addy, api_key):
        if False:
            for i in range(10):
                print('nop')
        self.ip = ip_addy
        self.api_key = api_key
        self.url = 'https://api.shodan.io/labs/honeyscore/{ip}?key={key}'
        self.headers = {'Referer': 'https://honeyscore.shodan.io/', 'Origin': 'https://honeyscore.shodan.io'}

    def make_request(self):
        if False:
            i = 10
            return i + 15
        try:
            req = requests.get(self.url.format(ip=self.ip, key=self.api_key), headers=self.headers)
            honeyscore = float(req.content)
        except Exception:
            honeyscore = 0.0
        return honeyscore