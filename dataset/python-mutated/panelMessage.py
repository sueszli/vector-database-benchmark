import os, sys, time
import public, json
if os.environ.get('BT_TASK') != '1':
    from BTPanel import cache
else:
    import cachelib
    cache = cachelib.SimpleCache()

class panelMessage:
    os = 'linux'

    def __init__(self):
        if False:
            return 10
        if not public.M('sqlite_master').where('type=? AND name=? AND sql LIKE ?', ('table', 'messages', '%retry_num%')).count():
            public.M('messages').execute('alter TABLE messages add send integer DEFAULT 0', ())
            public.M('messages').execute('alter TABLE messages add retry_num integer DEFAULT 0', ())
        pass

    def set_send_status(self, id, data):
        if False:
            while True:
                i = 10
        '\n            @name 设置消息发送状态\n            @author cjxin <2021-04-12>\n            @param args dict_obj{\n                id: 消息标识,\n                data\n            }\n            @return dict\n        '
        public.M('messages').where('id=?', id).update(data)
        return public.returnMsg(True, '设置成功!')
    '\n    获取官网推送消息，一天获取一次\n    '

    def get_cloud_messages(self, args):
        if False:
            return 10
        try:
            ret = cache.get('get_cloud_messages')
            if ret:
                return public.returnMsg(True, '同步成功1!')
            data = {}
            data['version'] = public.version()
            data['os'] = self.os
            sUrl = public.GetConfigValue('home') + '/api/wpanel/get_messages'
            import http_requests
            http_requests.DEFAULT_TYPE = 'src'
            info = http_requests.post(sUrl, data).json()
            for x in info:
                count = public.M('messages').where('level=? and msg=?', (x['level'], x['msg'])).count()
                if count:
                    continue
                pdata = {'level': x['level'], 'msg': x['msg'], 'state': 1, 'expire': int(time.time()) + int(x['expire']) * 86400, 'addtime': int(time.time())}
                public.M('messages').insert(pdata)
            cache.set('get_cloud_messages', 86400)
            return public.returnMsg(True, '同步成功!')
        except:
            return public.returnMsg(False, '同步失败!')

    def get_messages(self, args=None):
        if False:
            return 10
        '\n            @name 获取消息列表\n            @author hwliang <2020-05-18>\n            @return list\n        '
        public.run_thread(self.get_cloud_messages, args=(args,))
        data = public.M('messages').where('state=? and expire>?', (1, int(time.time()))).order('id desc').select()
        return data

    def get_messages_all(self, args=None):
        if False:
            return 10
        '\n            @name 获取所有消息列表\n            @author hwliang <2020-05-18>\n            @return list\n        '
        public.run_thread(self.get_cloud_messages, args=(args,))
        data = public.M('messages').order('id desc').select()
        return data

    def get_message_find(self, args=None, id=None):
        if False:
            print('Hello World!')
        '\n            @name 获取指定消息\n            @author hwliang <2020-05-18>\n            @param args dict_obj{\n                id: 消息标识\n            }\n            @return dict\n        '
        if args:
            id = int(args.id)
        data = public.M('messages').where('id=?', id).find()
        return data

    def create_message(self, args=None, level=None, msg=None, expire=None):
        if False:
            print('Hello World!')
        '\n            @name 创建新的消息\n            @author hwliang <2020-05-18>\n            @param args dict_obj{\n                level: 消息级别(info/warning/danger/error),\n                msg: 消息内容\n                expire: 过期时间\n            }\n            @return dict\n        '
        if args:
            level = args.level
            msg = args.msg
            expire = args.expire
        pdata = {'level': level, 'msg': msg, 'state': 1, 'expire': int(time.time()) + int(expire) * 86400, 'addtime': int(time.time())}
        public.M('messages').insert(pdata)
        return public.returnMsg(True, '创建成功!')

    def status_message(self, args=None, id=None, state=None):
        if False:
            print('Hello World!')
        '\n            @name 设置消息状态\n            @author hwliang <2020-05-18>\n            @param args dict_obj{\n                id: 消息标识,\n                state: 消息状态(0.已忽略, 1.正常)\n            }\n            @return dict\n        '
        if args:
            id = int(args.id)
            state = int(args.state)
        public.M('messages').where('id=?', id).setField('state', state)
        return public.returnMsg(True, '设置成功!')

    def remove_message(self, args=None, id=None):
        if False:
            i = 10
            return i + 15
        '\n            @name 删除指定消息\n            @author hwliang <2020-05-18>\n            @param args dict_obj{\n                id: 消息标识\n            }\n            @return dict\n        '
        if args:
            id = int(args.id)
        public.M('messages').where('id=?', id).delete()
        return public.returnMsg(True, '删除成功!')

    def remove_message_level(self, level):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 删除指定消息\n            @author hwliang <2020-05-18>\n            @param level string(指定级别或标识)\n            @return bool\n        '
        public.M('messages').where('(level=? or level=? or level=? or level=?) and state=?', (level, level + '15', level + '7', level + '3', 1)).delete()
        return True

    def remove_message_all(self):
        if False:
            for i in range(10):
                print('nop')
        public.M('messages').where('state=?', (1,)).delete()
        return True

    def is_level(self, level):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 指定消息是否忽略\n            @author hwliang <2020-05-18>\n            @param level string(指定级别或标识)\n            @return bool\n        '
        if public.M('messages').where('level=? and state=?', (level, 0)).count():
            return False
        else:
            return True