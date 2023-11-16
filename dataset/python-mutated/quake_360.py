import base64
import json
import time
import re
from app.services.dns_query import DNSQueryBase
from app import utils

class Query(DNSQueryBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Query, self).__init__()
        self.source_name = 'quake_360'
        self.api_url = 'https://quake.360.net/api/v3/search/quake_service'
        self.quake_token = None
        self.max_size = 500

    def init_key(self, quake_token=None, max_size=500):
        if False:
            for i in range(10):
                print('nop')
        self.quake_token = quake_token
        self.max_size = max_size

    def sub_domains(self, target):
        if False:
            i = 10
            return i + 15
        json_data = {'query': 'domain:"{}"'.format(target), 'include': ['service.http.host'], 'start': 0, 'size': self.max_size, 'latest': True}
        headers = {'X-QuakeToken': self.quake_token}
        conn = utils.http_req(self.api_url, 'post', json=json_data, headers=headers, timeout=(30.1, 100.1))
        if conn.status_code != 200:
            raise Exception('{}: {} QuakeToken is invalid'.format(self.source_name, self.quake_token))
        data = conn.json()
        if data['code'] != 0:
            raise Exception('{} error: {}'.format(self.source_name, json.dumps(data, ensure_ascii=False)))
        self.logger.debug('{}: target:{} meta:{}'.format(self.source_name, target, data['meta']))
        results = []
        items = data['data']
        for item in items:
            hostname = item['service']['http']['host']
            if hostname.endswith('.' + target):
                results.append(hostname)
        return list(set(results))