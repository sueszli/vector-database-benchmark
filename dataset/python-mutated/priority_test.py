from helpers import unittest
import luigi
import luigi.notifications
luigi.notifications.DEBUG = True

class PrioTask(luigi.Task):
    prio = luigi.Parameter()
    run_counter = 0

    @property
    def priority(self):
        if False:
            for i in range(10):
                print('nop')
        return self.prio

    def requires(self):
        if False:
            while True:
                i = 10
        if self.prio > 10:
            return PrioTask(self.prio - 10)

    def run(self):
        if False:
            return 10
        self.t = PrioTask.run_counter
        PrioTask.run_counter += 1

    def complete(self):
        if False:
            for i in range(10):
                print('nop')
        return hasattr(self, 't')

class PriorityTest(unittest.TestCase):

    def test_priority(self):
        if False:
            print('Hello World!')
        (p, q, r) = (PrioTask(1), PrioTask(2), PrioTask(3))
        luigi.build([p, q, r], local_scheduler=True)
        self.assertTrue(r.t < q.t < p.t)

    def test_priority_w_dep(self):
        if False:
            for i in range(10):
                print('nop')
        (x, y, z) = (PrioTask(25), PrioTask(15), PrioTask(5))
        (a, b, c) = (PrioTask(24), PrioTask(14), PrioTask(4))
        luigi.build([a, b, c, x, y, z], local_scheduler=True)
        self.assertTrue(z.t < y.t < x.t < c.t < b.t < a.t)