from . import SessionStorage

class SaeKVDBStorage(SessionStorage):
    """
    SaeKVDBStorage 使用SAE 的 KVDB 来保存你的session ::

        import werobot
        from werobot.session.saekvstorage import SaeKVDBStorage

        session_storage = SaeKVDBStorage()
        robot = werobot.WeRoBot(token="token", enable_session=True,
                                session_storage=session_storage)

    需要先在后台开启 KVDB 支持

    :param prefix: KVDB 中 Session 数据 key 的 prefix 。默认为 ``ws_``
    """

    def __init__(self, prefix='ws_'):
        if False:
            return 10
        try:
            import sae.kvdb
        except ImportError:
            raise RuntimeError('SaeKVDBStorage requires SAE environment')
        self.kv = sae.kvdb.KVClient()
        self.prefix = prefix

    def key_name(self, s):
        if False:
            print('Hello World!')
        return '{prefix}{s}'.format(prefix=self.prefix, s=s)

    def get(self, id):
        if False:
            i = 10
            return i + 15
        '\n        根据 id 获取数据。\n\n        :param id: 要获取的数据的 id\n        :return: 返回取到的数据，如果是空则返回一个空的 ``dict`` 对象\n        '
        return self.kv.get(self.key_name(id)) or {}

    def set(self, id, value):
        if False:
            print('Hello World!')
        '\n        根据 id 写入数据。\n\n        :param id: 要写入的 id\n        :param value: 要写入的数据，可以是一个 ``dict`` 对象\n        '
        return self.kv.set(self.key_name(id), value)

    def delete(self, id):
        if False:
            while True:
                i = 10
        '\n        根据 id 删除数据。\n\n        :param id: 要删除的数据的 id\n        '
        return self.kv.delete(self.key_name(id))