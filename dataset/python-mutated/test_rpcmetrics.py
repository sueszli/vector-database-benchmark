from queue import Queue
from threading import Thread
from sentry.services.hybrid_cloud.rpcmetrics import RpcMetricRecord, RpcMetricSpan, RpcMetricTracker
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class RpcMetricsTest(TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        assert len(RpcMetricTracker.get_local().spans) == 0

    def test_single_thread(self):
        if False:
            i = 10
            return i + 15
        with RpcMetricSpan() as span:
            for n in range(3):
                with RpcMetricRecord.measure(f'service{n}', f'method{n}'):
                    pass
            assert len(span.records) == 3

    def test_multithreaded(self):
        if False:
            while True:
                i = 10
        record_queue: Queue[RpcMetricRecord] = Queue()

        def make_thread(n: int) -> Thread:
            if False:
                while True:
                    i = 10

            def run() -> None:
                if False:
                    i = 10
                    return i + 15
                name = str(n)
                with RpcMetricSpan() as span:
                    with RpcMetricRecord.measure(name, name):
                        pass
                    assert len(span.records) == 1
                    (record,) = span.records
                    assert record.service_name == name
                    assert record.method_name == name
                    record_queue.put(record)
            return Thread(target=run)
        thread_count = 10
        threads = [make_thread(n) for n in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        records = list(record_queue.queue)
        assert len(records) == thread_count
        assert len({r.service_name for r in records}) == thread_count