from wechatpy.client.api.base import BaseWeChatAPI

class WeChatPoi(BaseWeChatAPI):
    """微信门店

    https://developers.weixin.qq.com/doc/offiaccount/WeChat_Stores/WeChat_Store_Interface.html
    """

    def add(self, poi_data):
        if False:
            i = 10
            return i + 15
        '\n        创建门店\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/WeChat_Stores/WeChat_Store_Interface.html#7\n\n        :param poi_data: 门店信息字典\n        :return: 返回的 JSON 数据包\n        '
        return self._post('poi/addpoi', data=poi_data)

    def get(self, poi_id):
        if False:
            i = 10
            return i + 15
        '\n        查询门店信息\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/WeChat_Stores/WeChat_Store_Interface.html#9\n\n        :param poi_id: 门店 ID\n        :return: 返回的 JSON 数据包\n        '
        return self._post('poi/getpoi', data={'poi_id': poi_id})

    def list(self, begin=0, limit=20):
        if False:
            for i in range(10):
                print('nop')
        '\n        查询门店列表\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/WeChat_Stores/WeChat_Store_Interface.html#10\n\n        :param begin: 开始位置，0 即为从第一条开始查询\n        :param limit: 返回数据条数，最大允许50，默认为20\n        :return: 返回的 JSON 数据包\n        '
        return self._post('poi/getpoilist', data={'begin': begin, 'limit': limit})

    def update(self, poi_data):
        if False:
            return 10
        '\n        修改门店服务信息\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/WeChat_Stores/WeChat_Store_Interface.html#11\n\n        :param poi_data: 门店信息字典\n        :return: 返回的 JSON 数据包\n        '
        return self._post('poi/updatepoi', data=poi_data)

    def delete(self, poi_id):
        if False:
            i = 10
            return i + 15
        '\n        删除门店\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/WeChat_Stores/WeChat_Store_Interface.html#12\n\n        :param poi_id: 门店 ID\n        :return: 返回的 JSON 数据包\n        '
        return self._post('poi/delpoi', data={'poi_id': poi_id})

    def get_categories(self):
        if False:
            return 10
        '\n        获取微信门店类目表\n\n        详情请参考\n        https://developers.weixin.qq.com/doc/offiaccount/WeChat_Stores/WeChat_Store_Interface.html#13\n\n        :return: 门店类目表\n        '
        res = self._get('api_getwxcategory', result_processor=lambda x: x['category_list'])
        return res