"""
测试celery
终端运行：celery -A python_celery:app worker -l INFO
"""
import time
from celery import Celery
broker = 'redis://localhost:6379/10'
backend = 'redis://localhost:6379/11'
app = Celery('tasks', broker=broker, backend=backend)

@app.task
def add(x, y):
    if False:
        for i in range(10):
            print('nop')
    print(time.time(), x, y)
    time.sleep(3)
    print(time.time(), x, y, '--')
    return x + y