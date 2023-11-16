from locust import User, task, constant
from locust.user.sequential_taskset import SequentialTaskSet
from locust.exception import RescheduleTask
from .testcases import LocustTestCase

class TestTaskSet(LocustTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()

        class MyUser(User):
            host = '127.0.0.1'
        self.locust = MyUser(self.environment)

    def test_task_sequence_with_list(self):
        if False:
            print('Hello World!')
        log = []

        def t1(ts):
            if False:
                print('Hello World!')
            log.append(1)

        def t2(ts):
            if False:
                for i in range(10):
                    print('nop')
            log.append(2)

        def t3(ts):
            if False:
                i = 10
                return i + 15
            log.append(3)
            ts.interrupt(reschedule=False)

        class MyTaskSequence(SequentialTaskSet):
            tasks = [t1, t2, t3]
        l = MyTaskSequence(self.locust)
        self.assertRaises(RescheduleTask, lambda : l.run())
        self.assertEqual([1, 2, 3], log)

    def test_task_sequence_with_methods(self):
        if False:
            for i in range(10):
                print('nop')
        log = []

        class MyTaskSequence(SequentialTaskSet):

            @task
            def t1(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(1)

            @task
            def t2(self):
                if False:
                    while True:
                        i = 10
                log.append(2)

            @task(1)
            def t3(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(3)
                self.interrupt(reschedule=False)
        l = MyTaskSequence(self.locust)
        self.assertRaises(RescheduleTask, lambda : l.run())
        self.assertEqual([1, 2, 3], log)

    def test_task_sequence_with_methods_and_list(self):
        if False:
            print('Hello World!')
        log = []

        def func_t1(ts):
            if False:
                for i in range(10):
                    print('nop')
            log.append(101)

        def func_t2(ts):
            if False:
                return 10
            log.append(102)

        class MyTaskSequence(SequentialTaskSet):

            @task
            def t1(self):
                if False:
                    i = 10
                    return i + 15
                log.append(1)

            @task
            def t2(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(2)
            tasks = [func_t1, func_t2]

            @task(1)
            def t3(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(3)
                self.interrupt(reschedule=False)
        l = MyTaskSequence(self.locust)
        self.assertRaises(RescheduleTask, lambda : l.run())
        self.assertEqual([1, 2, 101, 102, 3], log)

    def test_task_sequence_with_inheritance(self):
        if False:
            while True:
                i = 10
        log = []

        class TS1(SequentialTaskSet):

            @task
            def t1(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(1)
            tasks = [lambda ts: log.append(30)]

        class TS2(TS1):
            tasks = [lambda ts: log.append(20)]

            @task
            def t2(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(2)

        class TS3(TS2):

            @task
            def t3(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(3)
                self.interrupt(reschedule=False)
        l = TS3(self.locust)
        self.assertRaises(RescheduleTask, lambda : l.run())
        self.assertEqual([1, 30, 20, 2, 3], log)

    def test_task_sequence_multiple_iterations(self):
        if False:
            return 10
        log = []

        class TS(SequentialTaskSet):
            iteration_count = 0

            @task
            def t1(self):
                if False:
                    for i in range(10):
                        print('nop')
                log.append(1)

            @task
            def t2(self):
                if False:
                    i = 10
                    return i + 15
                log.append(2)

            @task(1)
            def t3(self):
                if False:
                    print('Hello World!')
                log.append(3)
                self.iteration_count += 1
                if self.iteration_count == 3:
                    self.interrupt(reschedule=False)
        l = TS(self.locust)
        self.assertRaises(RescheduleTask, lambda : l.run())
        self.assertEqual([1, 2, 3, 1, 2, 3, 1, 2, 3], log)