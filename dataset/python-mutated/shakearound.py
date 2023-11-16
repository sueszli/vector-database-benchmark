import time
from datetime import datetime
from optionaldict import optionaldict
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatShakeAround(BaseWeChatAPI):
    API_BASE_URL = 'https://api.weixin.qq.com/'

    @classmethod
    def _to_timestamp(cls, date):
        if False:
            i = 10
            return i + 15
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        if isinstance(date, datetime):
            timestamp = int(time.mktime(date.timetuple()))
            return timestamp
        return int(date)

    def apply_device_id(self, quantity, reason, poi_id=None, comment=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        申请设备ID\n        详情请参考\n        http://mp.weixin.qq.com/wiki/15/b9e012f917e3484b7ed02771156411f3.html\n\n        :param quantity: 申请的设备ID的数量，单次新增设备超过500个，需走人工审核流程\n        :param reason: 申请理由，不超过100个字\n        :param poi_id: 可选，设备关联的门店ID\n        :param comment: 可选，备注，不超过15个汉字或30个英文字母\n        :return: 申请的设备信息\n        '
        data = optionaldict()
        data['quantity'] = quantity
        data['apply_reason'] = reason
        data['poi_id'] = poi_id
        data['comment'] = comment
        res = self._post('shakearound/device/applyid', data=data, result_processor=lambda x: x['data'])
        return res

    def update_device(self, device_id=None, uuid=None, major=None, minor=None, comment=None):
        if False:
            print('Hello World!')
        '\n        更新设备信息\n        详情请参考\n        http://mp.weixin.qq.com/wiki/15/b9e012f917e3484b7ed02771156411f3.html\n\n        :param device_id: 设备编号，若填了UUID、major、minor，则可不填设备编号，若二者都填，则以设备编号为优先\n        :param uuid: UUID\n        :param major: major\n        :param minor: minor\n        :param comment: 设备的备注信息，不超过15个汉字或30个英文字母。\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict()
        data['comment'] = comment
        data['device_identifier'] = {'device_id': device_id, 'uuid': uuid, 'major': major, 'minor': minor}
        return self._post('shakearound/device/update', data=data)

    def bind_device_location(self, poi_id, device_id=None, uuid=None, major=None, minor=None):
        if False:
            while True:
                i = 10
        '\n        配置设备与门店的关联关系\n        详情请参考\n        http://mp.weixin.qq.com/wiki/15/b9e012f917e3484b7ed02771156411f3.html\n\n        :param poi_id: 待关联的门店ID\n        :param device_id: 设备编号，若填了UUID、major、minor，则可不填设备编号，若二者都填，则以设备编号为优先\n        :param uuid: UUID\n        :param major: major\n        :param minor: minor\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict()
        data['poi_id'] = poi_id
        data['device_identifier'] = {'device_id': device_id, 'uuid': uuid, 'major': major, 'minor': minor}
        return self._post('shakearound/device/bindlocation', data=data)

    def search_device(self, identifiers=None, apply_id=None, begin=0, count=10):
        if False:
            print('Hello World!')
        '\n        查询设备列表\n        详情请参考\n        http://mp.weixin.qq.com/wiki/15/b9e012f917e3484b7ed02771156411f3.html\n\n        :param identifiers: 设备 ID 信息列表\n        :param apply_id: 批次ID，申请设备ID超出500个时所返回批次ID\n        :param begin: 设备列表的起始索引值\n        :param count: 待查询的设备个数\n        :return: 设备列表\n        '
        data = optionaldict()
        data['begin'] = begin
        data['count'] = count
        data['apply_id'] = apply_id
        if identifiers:
            data['device_identifiers'] = identifiers
        res = self._post('shakearound/device/search', data=data, result_processor=lambda x: x['data'])
        return res

    def add_page(self, title, description, icon_url, page_url, comment=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        新增页面\n        详情请参考\n        http://mp.weixin.qq.com/wiki/5/6626199ea8757c752046d8e46cf13251.html\n\n        :param title: 在摇一摇页面展示的主标题，不超过6个字\n        :param description: 在摇一摇页面展示的副标题，不超过7个字\n        :param icon_url: 在摇一摇页面展示的图片。图片需先上传至微信侧服务器，\n                        用“素材管理-上传图片素材”接口上传图片，返回的图片URL再配置在此处\n        :param page_url: 跳转链接\n        :param comment: 可选，页面的备注信息，不超过15个字\n        :return: 页面信息\n        '
        data = optionaldict()
        data['title'] = title
        data['description'] = description
        data['icon_url'] = icon_url
        data['page_url'] = page_url
        data['comment'] = comment
        res = self._post('shakearound/page/add', data=data, result_processor=lambda x: x['data'])
        return res

    def update_page(self, page_id, title, description, icon_url, page_url, comment=None):
        if False:
            i = 10
            return i + 15
        '\n        编辑页面信息\n        详情请参考\n        http://mp.weixin.qq.com/wiki/5/6626199ea8757c752046d8e46cf13251.html\n\n        :param page_id: 摇周边页面唯一ID\n        :param title: 在摇一摇页面展示的主标题，不超过6个字\n        :param description: 在摇一摇页面展示的副标题，不超过7个字\n        :param icon_url: 在摇一摇页面展示的图片。图片需先上传至微信侧服务器，\n                        用“素材管理-上传图片素材”接口上传图片，返回的图片URL再配置在此处\n        :param page_url: 跳转链接\n        :param comment: 可选，页面的备注信息，不超过15个字\n        :return: 页面信息\n        '
        data = optionaldict()
        data['page_id'] = page_id
        data['title'] = title
        data['description'] = description
        data['icon_url'] = icon_url
        data['page_url'] = page_url
        data['comment'] = comment
        res = self._post('shakearound/page/update', data=data, result_processor=lambda x: x['data'])
        return res

    def search_pages(self, page_ids=None, begin=0, count=10):
        if False:
            print('Hello World!')
        '\n        查询页面列表\n        详情请参考\n        http://mp.weixin.qq.com/wiki/5/6626199ea8757c752046d8e46cf13251.html\n\n        :param page_ids: 指定页面的id列表\n        :param begin: 页面列表的起始索引值\n        :param count: 待查询的页面个数\n        :return: 页面查询结果信息\n        '
        if not page_ids:
            data = {'type': 2, 'begin': begin, 'count': count}
        else:
            if not isinstance(page_ids, (tuple, list)):
                page_ids = [page_ids]
            data = {'type': 1, 'page_ids': page_ids}
        res = self._post('shakearound/page/search', data=data, result_processor=lambda x: x['data'])
        return res

    def delete_page(self, page_id):
        if False:
            print('Hello World!')
        '\n        删除页面\n        详情请参考\n        http://mp.weixin.qq.com/wiki/5/6626199ea8757c752046d8e46cf13251.html\n\n        :param page_id: 指定页面的id列表\n        :return: 返回的 JSON 数据包\n        '
        return self._post('shakearound/page/delete', data={'page_id': page_id})

    def add_material(self, media_file, media_type='icon'):
        if False:
            print('Hello World!')
        '\n        上传图片素材\n        详情请参考\n        http://mp.weixin.qq.com/wiki/5/e997428269ff189d8f9a4b9e177be2d9.html\n\n        :param media_file: 要上传的文件，一个 File-object\n        :param media_type: 摇一摇素材类型, 取值为 icon或者 license, 默认 icon.\n        :return: 上传的素材信息\n        '
        res = self._post('shakearound/material/add', files={'media': media_file}, params={'type': media_type}, result_processor=lambda x: x['data'])
        return res

    def bind_device_pages(self, page_ids, bind, append, device_id=None, uuid=None, major=None, minor=None):
        if False:
            while True:
                i = 10
        '\n        配置设备与页面的关联关系\n        详情请参考\n        http://mp.weixin.qq.com/wiki/12/c8120214ec0ba08af5dfcc0da1a11400.html\n\n        :param page_ids: 待关联的页面列表\n        :param bind: 关联操作标志位， 0为解除关联关系，1为建立关联关系\n        :param append: 新增操作标志位， 0为覆盖，1为新增\n        :param device_id: 设备编号，若填了UUID、major、minor，则可不填设备编号，若二者都填，则以设备编号为优先\n        :param uuid: UUID\n        :param major: major\n        :param minor: minor\n        :return: 返回的 JSON 数据包\n        '
        if not isinstance(page_ids, (tuple, list)):
            page_ids = [page_ids]
        data = {'page_ids': page_ids, 'bind': int(bind), 'append': int(append), 'device_identifier': {'device_id': device_id, 'uuid': uuid, 'major': major, 'minor': minor}}
        return self._post('shakearound/device/bindpage', data=data)

    def get_shake_info(self, ticket):
        if False:
            while True:
                i = 10
        '\n        获取摇周边的设备及用户信息\n        详情请参考\n        http://mp.weixin.qq.com/wiki/3/34904a5db3d0ec7bb5306335b8da1faf.html\n\n        :param ticket: 摇周边业务的ticket，可在摇到的URL中得到，ticket生效时间为30分钟\n        :return: 设备及用户信息\n        '
        res = self._post('shakearound/user/getshakeinfo', data={'ticket': ticket}, result_processor=lambda x: x['data'])
        return res

    def get_device_statistics(self, begin_date, end_date, device_id=None, uuid=None, major=None, minor=None):
        if False:
            while True:
                i = 10
        '\n        以设备为维度的数据统计接口\n        http://mp.weixin.qq.com/wiki/0/8a24bcacad40fe7ee98d1573cb8a6764.html\n\n        :param begin_date: 起始时间，最长时间跨度为30天\n        :param end_date: 结束时间，最长时间跨度为30天\n        :param device_id: 设备编号，若填了UUID、major、minor，则可不填设备编号，若二者都填，则以设备编号为优先\n        :param uuid: UUID\n        :param major: major\n        :param minor: minor\n        '
        data = {'device_identifier': {'device_id': device_id, 'uuid': uuid, 'major': major, 'minor': minor}, 'begin_date': self._to_timestamp(begin_date), 'end_date': self._to_timestamp(end_date)}
        res = self._post('shakearound/statistics/device', data=data, result_processor=lambda x: x['data'])
        return res

    def get_page_statistics(self, page_id, begin_date, end_date):
        if False:
            while True:
                i = 10
        '\n        以页面为维度的数据统计接口\n        详情请参考\n        http://mp.weixin.qq.com/wiki/0/8a24bcacad40fe7ee98d1573cb8a6764.html\n\n        :param page_id: 页面 ID\n        :param begin_date: 起始时间，最长时间跨度为30天\n        :param end_date: 结束时间，最长时间跨度为30天\n        :return: 统计数据\n        '
        res = self._post('shakearound/statistics/page', data={'page_id': page_id, 'begin_date': self._to_timestamp(begin_date), 'end_date': self._to_timestamp(end_date)}, result_processor=lambda x: x['data'])
        return res

    def get_apply_status(self, apply_id):
        if False:
            while True:
                i = 10
        '\n        查询设备ID申请审核状态\n        详情请参考\n        http://mp.weixin.qq.com/wiki/15/b9e012f917e3484b7ed02771156411f3.html\n\n        :param apply_id: 批次ID，申请设备ID时所返回的批次ID\n        :return: 批次状态信息\n        '
        res = self._post('shakearound/device/applystatus', data={'apply_id': apply_id}, result_processor=lambda x: x['data'])
        return res