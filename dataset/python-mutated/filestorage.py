try:
    import anydbm as dbm
    assert dbm
except ImportError:
    import dbm
from werobot.session import SessionStorage
from werobot.utils import json_loads, json_dumps, to_binary

class FileStorage(SessionStorage):
    """
    FileStorage 会把你的 Session 数据以 dbm 形式储存在文件中。

    :param filename: 文件名， 默认为 ``werobot_session``
    """

    def __init__(self, filename: str='werobot_session'):
        if False:
            return 10
        try:
            self.db = dbm.open(filename, 'c')
        except TypeError:
            self.db = dbm.open(to_binary(filename), 'c')

    def get(self, id):
        if False:
            print('Hello World!')
        '\n        根据 id 获取数据。\n\n        :param id: 要获取的数据的 id\n        :return: 返回取到的数据，如果是空则返回一个空的 ``dict`` 对象\n        '
        try:
            session_json = self.db[id]
        except KeyError:
            session_json = '{}'
        return json_loads(session_json)

    def set(self, id, value):
        if False:
            return 10
        '\n        根据 id 写入数据。\n\n        :param id: 要写入的 id\n        :param value: 要写入的数据，可以是一个 ``dict`` 对象\n        '
        self.db[id] = json_dumps(value)

    def delete(self, id):
        if False:
            print('Hello World!')
        '\n        根据 id 删除数据。\n\n        :param id: 要删除的数据的 id\n        '
        del self.db[id]