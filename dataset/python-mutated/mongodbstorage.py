from werobot.session import SessionStorage
from werobot.utils import json_loads, json_dumps

class MongoDBStorage(SessionStorage):
    """
    MongoDBStorage 会把你的 Session 数据储存在一个 MongoDB Collection 中 ::

        import pymongo
        import werobot
        from werobot.session.mongodbstorage import MongoDBStorage

        collection = pymongo.MongoClient()["wechat"]["session"]
        session_storage = MongoDBStorage(collection)
        robot = werobot.WeRoBot(token="token", enable_session=True,
                                session_storage=session_storage)


    你需要安装 ``pymongo`` 才能使用 MongoDBStorage 。

    :param collection: 一个 MongoDB Collection。
    """

    def __init__(self, collection):
        if False:
            while True:
                i = 10
        self.collection = collection
        collection.create_index('wechat_id')

    def _get_document(self, id):
        if False:
            print('Hello World!')
        return self.collection.find_one({'wechat_id': id})

    def get(self, id):
        if False:
            return 10
        '\n        根据 id 获取数据。\n\n        :param id: 要获取的数据的 id\n        :return: 返回取到的数据，如果是空则返回一个空的 ``dict`` 对象\n        '
        document = self._get_document(id)
        if document:
            session_json = document['session']
            return json_loads(session_json)
        return {}

    def set(self, id, value):
        if False:
            print('Hello World!')
        '\n        根据 id 写入数据。\n\n        :param id: 要写入的 id\n        :param value: 要写入的数据，可以是一个 ``dict`` 对象\n                '
        session = json_dumps(value)
        self.collection.replace_one({'wechat_id': id}, {'wechat_id': id, 'session': session}, upsert=True)

    def delete(self, id):
        if False:
            while True:
                i = 10
        '\n        根据 id 删除数据。\n\n        :param id: 要删除的数据的 id\n        '
        self.collection.delete_one({'wechat_id': id})