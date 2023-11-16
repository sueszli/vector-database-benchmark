from wechatpy.client.api.base import BaseWeChatAPI

class MerchantExpress(BaseWeChatAPI):
    API_BASE_URL = 'https://api.weixin.qq.com/'

    def add(self, delivery_template):
        if False:
            while True:
                i = 10
        return self._post('merchant/express/add', data={'delivery_template': delivery_template})

    def delete(self, template_id):
        if False:
            i = 10
            return i + 15
        return self._post('merchant/express/del', data={'template_id': template_id})

    def update(self, template_id, delivery_template):
        if False:
            for i in range(10):
                print('nop')
        return self._post('merchant/express/update', data={'template_id': template_id, 'delivery_template': delivery_template})

    def get(self, template_id):
        if False:
            while True:
                i = 10
        res = self._post('merchant/express/getbyid', data={'template_id': template_id}, result_processor=lambda x: x['template_info'])
        return res

    def get_all(self):
        if False:
            for i in range(10):
                print('nop')
        res = self._get('merchant/express/getall', result_processor=lambda x: x['template_info'])
        return res