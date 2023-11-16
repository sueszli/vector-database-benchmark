from kombu import Queue
from celery.utils.nodenames import worker_direct

class test_worker_direct:

    def test_returns_if_queue(self):
        if False:
            for i in range(10):
                print('nop')
        q = Queue('foo')
        assert worker_direct(q) is q