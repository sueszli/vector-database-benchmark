from gevent import monkey
monkey.patch_all()
import gevent
from huey.contrib.mini import MiniHuey
huey = MiniHuey()
huey.start()

@huey.task()
def add(a, b):
    if False:
        print('Hello World!')
    return a + b
res = add(1, 2)
print(res())
print('Scheduling task for execution in 2 seconds.')
res = add.schedule(args=(10, 20), delay=2)
print(res())
huey.stop()