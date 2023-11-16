from werkzeug.contrib.cache import FileSystemCache

class WechatCache(FileSystemCache):
    """基于文件的缓存

    """

    def __init__(self, cache_dir='/tmp/wechatsogou-cache', default_timeout=300):
        if False:
            for i in range(10):
                print('nop')
        '初始化\n\n        cache_dir是缓存目录\n        '
        super(WechatCache, self).__init__(cache_dir, default_timeout)

    def get(self, key):
        if False:
            while True:
                i = 10
        try:
            return super(WechatCache, self).get(key)
        except ValueError:
            return None