import gevent
from gevent import testing as greentest

def worker(i):
    if False:
        while True:
            i = 10
    if i == 2:
        raise ValueError(i)
    return i

class Test(greentest.TestCase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        finished = 0
        done_worker = gevent.spawn(worker, 'done')
        gevent.joinall((done_worker,))
        workers = [gevent.spawn(worker, i) for i in range(3)]
        workers.append(done_worker)
        for _ in gevent.iwait(workers):
            finished += 1
            try:
                gevent.sleep(0.01)
            except ValueError as ex:
                self.assertEqual(ex.args[0], 2)
        self.assertEqual(finished, 4)
if __name__ == '__main__':
    greentest.main()