import os
import time
import threading
import unittest
from pyspark import SparkContext, SparkConf, InheritableThread

class PinThreadTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.old_pin_thread = os.environ.get('PYSPARK_PIN_THREAD')
        os.environ['PYSPARK_PIN_THREAD'] = 'true'
        cls.sc = SparkContext('local[4]', cls.__name__, conf=SparkConf())

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        cls.sc.stop()
        if cls.old_pin_thread is not None:
            os.environ['PYSPARK_PIN_THREAD'] = cls.old_pin_thread
        else:
            del os.environ['PYSPARK_PIN_THREAD']

    def test_pinned_thread(self):
        if False:
            for i in range(10):
                print('nop')
        threads = []
        exceptions = []
        property_name = 'test_property_%s' % PinThreadTests.__name__
        jvm_thread_ids = []
        for i in range(10):

            def test_local_property():
                if False:
                    return 10
                jvm_thread_id = self.sc._jvm.java.lang.Thread.currentThread().getId()
                jvm_thread_ids.append(jvm_thread_id)
                self.sc.setLocalProperty(property_name, str(i))
                time.sleep(i % 2)
                try:
                    assert self.sc.getLocalProperty(property_name) == str(i)
                    assert jvm_thread_id == self.sc._jvm.java.lang.Thread.currentThread().getId()
                except Exception as e:
                    exceptions.append(e)
            threads.append(threading.Thread(target=test_local_property))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for e in exceptions:
            raise e
        assert len(set(jvm_thread_ids)) == 10

    def test_multiple_group_jobs(self):
        if False:
            return 10
        self.check_job_cancellation(lambda job_group: self.sc.setJobGroup(job_group, 'test rdd collect with setting job group'), lambda job_group: self.sc.cancelJobGroup(job_group))

    def test_multiple_group_tags(self):
        if False:
            while True:
                i = 10
        self.check_job_cancellation(lambda job_tag: self.sc.addJobTag(job_tag), lambda job_tag: self.sc.cancelJobsWithTag(job_tag))

    def check_job_cancellation(self, setter, canceller):
        if False:
            return 10
        job_id_a = 'job_ids_to_cancel'
        job_id_b = 'job_ids_to_run'
        threads = []
        thread_ids = range(4)
        thread_ids_to_cancel = [i for i in thread_ids if i % 2 == 0]
        thread_ids_to_run = [i for i in thread_ids if i % 2 != 0]
        is_job_cancelled = [False for _ in thread_ids]

        def run_job(job_id, index):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Executes a job with the group ``job_group``. Each job waits for 3 seconds\n            and then exits.\n            '
            try:
                setter(job_id)
                self.sc.parallelize([15]).map(lambda x: time.sleep(x)).collect()
                is_job_cancelled[index] = False
            except Exception:
                is_job_cancelled[index] = True
        run_job(job_id_a, 0)
        self.assertFalse(is_job_cancelled[0])
        for i in thread_ids_to_cancel:
            t = threading.Thread(target=run_job, args=(job_id_a, i))
            t.start()
            threads.append(t)
        for i in thread_ids_to_run:
            t = threading.Thread(target=run_job, args=(job_id_b, i))
            t.start()
            threads.append(t)
        time.sleep(3)
        canceller(job_id_a)
        for t in threads:
            t.join()
        for i in thread_ids_to_cancel:
            self.assertTrue(is_job_cancelled[i], 'Thread {i}: Job in group A was not cancelled.'.format(i=i))
        for i in thread_ids_to_run:
            self.assertFalse(is_job_cancelled[i], 'Thread {i}: Job in group B did not succeeded.'.format(i=i))

    def test_inheritable_local_property(self):
        if False:
            print('Hello World!')
        self.sc.setLocalProperty('a', 'hi')
        expected = []

        def get_inner_local_prop():
            if False:
                while True:
                    i = 10
            expected.append(self.sc.getLocalProperty('b'))

        def get_outer_local_prop():
            if False:
                i = 10
                return i + 15
            expected.append(self.sc.getLocalProperty('a'))
            self.sc.setLocalProperty('b', 'hello')
            t2 = InheritableThread(target=get_inner_local_prop)
            t2.start()
            t2.join()
        t1 = InheritableThread(target=get_outer_local_prop)
        t1.start()
        t1.join()
        self.assertEqual(self.sc.getLocalProperty('b'), None)
        self.assertEqual(expected, ['hi', 'hello'])
if __name__ == '__main__':
    import unittest
    from pyspark.tests.test_pin_thread import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)