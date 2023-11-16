import os
import queue
import sys
import threading
import time
taskQueue = queue.Queue()
lock = threading.RLock()

def worker(fun):
    if False:
        i = 10
        return i + 15
    while True:
        temp = taskQueue.get()
        fun(temp)
        taskQueue.task_done()

def threadPool(threadPoolNum):
    if False:
        for i in range(10):
            print('nop')
    threadPool = []
    for i in range(threadPoolNum):
        thread = threading.Thread(target=worker, args={doFun})
        thread.daemon = True
        threadPool.append(thread)
    return threadPool

def getPyCovResult(params):
    if False:
        print('Hello World!')
    rootPath = params[0]
    ut = params[1]
    print('ut: %s' % ut)
    startTime = int(time.time())
    path = f'{rootPath}/build/pytest/{ut}'
    os.system('cd %s && coverage combine `ls python-coverage.data.*`' % path)
    os.system('cd %s && pwd && coverage xml -i -o python-coverage.xml' % path)
    xml_path = '%s/python-coverage.xml' % path
    os.system(f'python2.7 {rootPath}/tools/analysisPyXml.py {rootPath} {ut}')
    endTime = int(time.time())
    print('pyCov Time: %s' % (endTime - startTime))

def doFun(params):
    if False:
        return 10
    getPyCovResult(params)

def main(rootPath):
    if False:
        while True:
            i = 10
    '\n    1. get gcov file\n    2. get gcov file not coverageratio = 0\n    '
    path = '%s/build/pytest' % rootPath
    dirs = os.listdir(path)
    pool = threadPool(23)
    for i in range(pool.__len__()):
        pool[i].start()
    for ut in dirs:
        params = [rootPath, ut]
        taskQueue.put(params)
    taskQueue.join()
if __name__ == '__main__':
    rootPath = sys.argv[1]
    main(rootPath)