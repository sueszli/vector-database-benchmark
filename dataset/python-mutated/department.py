from itertools import chain
from optionaldict import optionaldict
from wechatpy.client.api.base import BaseWeChatAPI

class WeChatDepartment(BaseWeChatAPI):

    def create(self, name, parent_id=1, order=None, id=None, name_en=None):
        if False:
            i = 10
            return i + 15
        '\n        创建部门\n\n        详情请参考\n        https://developer.work.weixin.qq.com/document/path/90205\n\n        :param name: 部门名称。长度限制为1~32个字符，字符不能包括\\:?”<>｜\n        :param parent_id: 父部门id，32位整型\n        :param order: 在父部门中的次序值。order值大的排序靠前。有效的值范围是[0, 2^32)\n        :param id: 部门id，32位整型，指定时必须大于1。若不填该参数，将自动生成id\n        :param name_en: 英文名称。同一个层级的部门名称不能重复。需要在管理后台开启多语言支持才能生效。长度限制为1~32个字符，字符不能包括:*?"<>｜\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict(name=name, parentid=parent_id, order=order, id=id, name_en=name_en)
        return self._post('department/create', data=data)

    def update(self, id, name=None, parent_id=None, order=None, name_en=None):
        if False:
            while True:
                i = 10
        '\n        更新部门\n\n        详情请参考\n        https://developer.work.weixin.qq.com/document/path/90206\n\n        :param id: 部门 id\n        :param name: 部门名称。长度限制为1~32个字符，字符不能包括\\:?”<>｜\n        :param parent_id: 父亲部门id\n        :param order: 在父部门中的次序值。order值大的排序靠前。有效的值范围是[0, 2^32)\n        :param name_en: 英文名称。同一个层级的部门名称不能重复。需要在管理后台开启多语言支持才能生效。长度限制为1~32个字符，字符不能包括:*?"<>｜\n        :return: 返回的 JSON 数据包\n        '
        data = optionaldict(id=id, name=name, parentid=parent_id, order=order, name_en=name_en)
        return self._post('department/update', data=data)

    def delete(self, id):
        if False:
            print('Hello World!')
        '\n        删除部门\n\n        详情请参考\n        https://developer.work.weixin.qq.com/document/path/90207\n\n        :param id: 部门id。（注：不能删除根部门；不能删除含有子部门、成员的部门）\n        :return: 返回的 JSON 数据包\n        '
        return self._get('department/delete', params={'id': id})

    def list(self, id=None):
        if False:
            while True:
                i = 10
        '\n        获取指定部门列表\n\n        详情请参考\n        https://developer.work.weixin.qq.com/document/path/90208\n\n        权限说明：\n        只能拉取token对应的应用的权限范围内的部门列表\n\n        :param id: 部门id。获取指定部门及其下的子部门。 如果不填，默认获取全量组织架构\n        :return: 部门列表\n        '
        if id is None:
            res = self._get('department/list')
        else:
            res = self._get('department/list', params={'id': id})
        return res['department']

    def simple_list(self, id=None):
        if False:
            return 10
        '\n        获取子部门 ID 列表，和 list 接口相比，此接口只返回部门 ID，ORDER 和 PARENTID 字段\n\n        详情请参考\n        https://developer.work.weixin.qq.com/document/path/95350\n\n        :param id: 部门id。获取指定部门及其下的子部门（以及子部门的子部门等等，递归）。 如果不填，默认获取全量组织架构\n        :return: 部门列表\n        '
        if id is None:
            res = self._get('department/simplelist')
        else:
            res = self._get('department/simplelist', params={'id': id})
        return res['department_id']

    def get(self, id):
        if False:
            while True:
                i = 10
        '\n        获取单个部门详情\n\n        详情请参考\n        https://developer.work.weixin.qq.com/document/path/95351\n\n        :param id: 部门 ID\n        :return: 部门信息\n        '
        res = self._get('department/get', params={'id': id})
        return res['department']

    def get_users(self, id, fetch_child=0, simple=True):
        if False:
            while True:
                i = 10
        '\n        获取部门成员：https://developer.work.weixin.qq.com/document/path/90200\n\n        获取部门成员详情：https://developer.work.weixin.qq.com/document/path/90201\n\n        :param id: 部门 id\n        :param fetch_child: 1/0：是否递归获取子部门下面的成员\n        :param simple: True 获取部门成员，False 获取部门成员详情\n        :return: 部门成员列表\n        '
        url = 'user/simplelist' if simple else 'user/list'
        res = self._get(url, params={'department_id': id, 'fetch_child': 1 if fetch_child else 0})
        return res['userlist']

    def get_map_users(self, id=None, key='name', fetch_child=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        映射员工某详细字段到 ``user_id``\n\n        企业微信许多对员工操作依赖于 ``user_id`` ，但没有提供直接查询员工对应 ``user_id`` 的结构，\n\n        这里是一个变通的方法，常用于储存员工 ``user_id`` ，并用于后续查询或对单人操作（如发送指定消息）\n\n        :param id: 部门 id， 如果不填，默认获取有权限的所有部门\n        :param key: 员工详细信息字段 key，所指向的值必须唯一\n        :param fetch_child: 1/0：是否递归获取子部门下面的成员\n        :return: dict - 部门成员指定字段到 user_id 的 map  ``{ key: user_id }``\n        '
        ids = [id] if id is not None else [item['id'] for item in self.list()]
        users_info = list(chain(*[self.get_users(department, fetch_child=fetch_child, simple=False) for department in ids]))
        users_zip = [(user[key], user['userid']) for user in users_info]
        return dict(users_zip)