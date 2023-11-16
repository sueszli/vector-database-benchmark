from typing import List
from optionaldict import optionaldict
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatExternalContactGroupChat(BaseWeChatAPI):
    """
    https://work.weixin.qq.com/api/doc#90000/90135/90221
    """

    def list(self, status_filter: int=0, owner_userid_list: List=None, cursor: str=None, limit: int=100):
        if False:
            return 10
        '\n        该接口用于获取配置过客户群管理的客户群列表。\n        https://work.weixin.qq.com/api/doc/90000/90135/92120\n        :return: 返回的 JSON 数据包\n        :param status_filter: 客户群跟进状态过滤。[0 - 所有列表(即不过滤)\n                                                1 - 离职待继承\n                                                2 - 离职继承中\n                                                3 - 离职继承完成\n\n                                                默认为0]\n        :param owner_userid_list: 根据群主id过滤。如果不填，表示获取应用可见范围内全部群主的数据（但是不建议这么用，如果可见范围人数超过1000人，为了防止数据包过大，会报错 81017）\n        :param cursor: 用于分页查询的游标，字符串类型，由上一次调用返回，首次调用不填\n        :param limit: 分页，预期请求的数据量，取值范围 1 ~ 1000， 默认100\n        :return:\n        '
        data = optionaldict(status_filter=status_filter, cursor=cursor, limit=limit)
        if owner_userid_list:
            data['owner_filter'] = {'userid_list': owner_userid_list}
        return self._post('externalcontact/groupchat/list', data=data)

    def list_all(self, status_filter: int=0, owner_userid_list: List=None, limit: int=100) -> List:
        if False:
            print('Hello World!')
        '\n        该接口用于获取配置过客户群管理的所有客户群列表，自动走完所有分页\n        '
        chat_list = []
        cursor = None
        while True:
            result = self.list(status_filter, owner_userid_list, cursor, limit)
            if result['errcode'] == 0:
                chat_list.extend(result['group_chat_list'])
            if result.get('next_cursor'):
                cursor = result['next_cursor']
            else:
                break
        return chat_list

    def get(self, chat_id: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        通过客户群ID，获取详情。包括群名、群成员列表、群成员入群时间、入群方式。（客户群是由具有客户群使用权限的成员创建的外部群）\n        需注意的是，如果发生群信息变动，会立即收到群变更事件，但是部分信息是异步处理，可能需要等一段时间调此接口才能得到最新结果\n        PS: 接口命名为get，调用却是POST，无语\n        https://work.weixin.qq.com/api/doc/90000/90135/92122\n        :param chat_id: 客户群ID\n        :return: 返回的 JSON 数据包\n        '
        return self._post('externalcontact/groupchat/get', data={'chat_id': chat_id})

    def statistic(self, day_begin_time: int, day_end_time: int=None, owner_userid_list: List=None, order_by: int=1, order_asc: int=0, offset: int=0, limit: int=500):
        if False:
            return 10
        '\n        获取指定日期的统计数据。注意，企业微信仅存储180天的数据。\n        :param day_begin_time: 起始日期的时间戳，填当天的0时0分0秒（否则系统自动处理为当天的0分0秒）。取值范围：昨天至前180天。\n        :param day_end_time: 结束日期的时间戳，填当天的0时0分0秒（否则系统自动处理为当天的0分0秒）。取值范围：昨天至前180天。\n                            如果不填，默认同 day_begin_time（即默认取一天的数据）\n        :param owner_userid_list: 群主过滤，如果不填，表示获取全部群主的数据\n        :param order_by:    排序方式。默认为1\n                            [1 - 新增群的数量\n                            2 - 群总数\n                            3 - 新增群人数\n                            4 - 群总人数]\n        :param order_asc: 是否升序。0-否；1-是。默认降序,即0\n        :param offset: 分页，偏移量, 默认为0\n        :param limit: 分页，预期请求的数据量，默认为500，取值范围 1 ~ 1000\n        :return: 返回的 JSON 数据包\n        '
        if not day_end_time:
            day_end_time = day_begin_time
        data = optionaldict(day_begin_time=day_begin_time, day_end_time=day_end_time, order_by=order_by, order_asc=order_asc, offset=offset, limit=limit)
        if owner_userid_list:
            data['owner_filter'] = {'userid_list': owner_userid_list}
        return self._post('externalcontact/groupchat/statistic', data=data)