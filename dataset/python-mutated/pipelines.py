import pymongo

class WechatprojectPipeline(object):

    def __init__(self):
        if False:
            return 10
        connection = pymongo.Connection(host='localhost', port=27017)
        db = connection['testwechat']
        self.posts = db['result']

    def process_item(self, item, spider):
        if False:
            print('Hello World!')
        self.posts.insert(dict(item))
        return item