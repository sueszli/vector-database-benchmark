"""
调度的使用
"""
import time
import datetime
from threading import Timer
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler

def print_hello():
    if False:
        i = 10
        return i + 15
    print('TimeNow in func: %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return
if __name__ == '__main__':
    scheduler = BlockingScheduler()
    scheduler.add_job(print_hello, 'interval', seconds=5)
    scheduler = BackgroundScheduler()
    scheduler.add_job(print_hello, 'interval', seconds=5)