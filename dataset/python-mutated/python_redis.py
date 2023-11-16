"""
Python操作Redis实现消息的发布与订阅
"""
import sys
import time
import redis
conn_pool = redis.ConnectionPool(host='localhost', port=6379, db=1)
conn_inst = redis.Redis(connection_pool=conn_pool)
channel_name = 'fm-101.1'

def public_test():
    if False:
        print('Hello World!')
    while True:
        conn_inst.publish(channel_name, 'hello ' + str(time.time()))
        if int(time.time()) % 10 == 1:
            conn_inst.publish(channel_name, 'over')
        time.sleep(1)

def subscribe_test(_type=0):
    if False:
        return 10
    pub = conn_inst.pubsub()
    pub.subscribe(channel_name)
    if _type == 0:
        for item in pub.listen():
            print('Listen on channel: %s' % item)
            if item['type'] == 'message' and item['data'].decode() == 'over':
                print(item['channel'].decode(), '已停止发布')
                break
    else:
        while True:
            item = pub.parse_response()
            print('Listen on channel: %s' % item)
            if item[0].decode() == 'message' and item[2].decode() == 'over':
                print(item[1].decode(), '已停止发布')
                break
    pub.unsubscribe()
    return
if __name__ == '__main__':
    if sys.argv[1] == 'public':
        public_test()
    else:
        subscribe_test(int(sys.argv[2]))