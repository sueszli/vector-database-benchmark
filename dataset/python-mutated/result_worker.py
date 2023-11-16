import time
import json
import logging
from six.moves import queue as Queue
logger = logging.getLogger('result')

class ResultWorker(object):
    """
    do with result
    override this if needed.
    """

    def __init__(self, resultdb, inqueue):
        if False:
            return 10
        self.resultdb = resultdb
        self.inqueue = inqueue
        self._quit = False

    def on_result(self, task, result):
        if False:
            print('Hello World!')
        'Called every result'
        if not result:
            return
        if 'taskid' in task and 'project' in task and ('url' in task):
            logger.info('result %s:%s %s -> %.30r' % (task['project'], task['taskid'], task['url'], result))
            return self.resultdb.save(project=task['project'], taskid=task['taskid'], url=task['url'], result=result)
        else:
            logger.warning('result UNKNOW -> %.30r' % result)
            return

    def quit(self):
        if False:
            while True:
                i = 10
        self._quit = True

    def run(self):
        if False:
            print('Hello World!')
        'Run loop'
        logger.info('result_worker starting...')
        while not self._quit:
            try:
                (task, result) = self.inqueue.get(timeout=1)
                self.on_result(task, result)
            except Queue.Empty as e:
                continue
            except KeyboardInterrupt:
                break
            except AssertionError as e:
                logger.error(e)
                continue
            except Exception as e:
                logger.exception(e)
                continue
        logger.info('result_worker exiting...')

class OneResultWorker(ResultWorker):
    """Result Worker for one mode, write results to stdout"""

    def on_result(self, task, result):
        if False:
            i = 10
            return i + 15
        'Called every result'
        if not result:
            return
        if 'taskid' in task and 'project' in task and ('url' in task):
            logger.info('result %s:%s %s -> %.30r' % (task['project'], task['taskid'], task['url'], result))
            print(json.dumps({'taskid': task['taskid'], 'project': task['project'], 'url': task['url'], 'result': result, 'updatetime': time.time()}))
        else:
            logger.warning('result UNKNOW -> %.30r' % result)
            return