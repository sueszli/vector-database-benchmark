import gevent
import gevent.testing as greentest
from gevent.lock import Semaphore

class Testiwait(greentest.TestCase):

    def test_noiter(self):
        if False:
            for i in range(10):
                print('nop')
        sem1 = Semaphore()
        sem2 = Semaphore()
        gevent.spawn(sem1.release)
        ready = next(gevent.iwait((sem1, sem2)))
        self.assertEqual(sem1, ready)

    def test_iwait_partial(self):
        if False:
            while True:
                i = 10
        sem = Semaphore()
        let = gevent.spawn(sem.release)
        with gevent.iwait((sem,), timeout=0.01) as iterator:
            self.assertEqual(sem, next(iterator))
        let.get()

    def test_iwait_nogarbage(self):
        if False:
            i = 10
            return i + 15
        sem1 = Semaphore()
        sem2 = Semaphore()
        let = gevent.spawn(sem1.release)
        with gevent.iwait((sem1, sem2)) as iterator:
            self.assertEqual(sem1, next(iterator))
            self.assertEqual(sem2.linkcount(), 1)
        self.assertEqual(sem2.linkcount(), 0)
        let.get()
if __name__ == '__main__':
    greentest.main()