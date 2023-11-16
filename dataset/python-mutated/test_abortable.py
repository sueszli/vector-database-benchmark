from celery.contrib.abortable import AbortableAsyncResult, AbortableTask

class test_AbortableTask:

    def setup_method(self):
        if False:
            while True:
                i = 10

        @self.app.task(base=AbortableTask, shared=False)
        def abortable():
            if False:
                while True:
                    i = 10
            return True
        self.abortable = abortable

    def test_async_result_is_abortable(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.abortable.apply_async()
        tid = result.id
        assert isinstance(self.abortable.AsyncResult(tid), AbortableAsyncResult)

    def test_is_not_aborted(self):
        if False:
            while True:
                i = 10
        self.abortable.push_request()
        try:
            result = self.abortable.apply_async()
            tid = result.id
            assert not self.abortable.is_aborted(task_id=tid)
        finally:
            self.abortable.pop_request()

    def test_is_aborted_not_abort_result(self):
        if False:
            i = 10
            return i + 15
        self.abortable.AsyncResult = self.app.AsyncResult
        self.abortable.push_request()
        try:
            self.abortable.request.id = 'foo'
            assert not self.abortable.is_aborted()
        finally:
            self.abortable.pop_request()

    def test_abort_yields_aborted(self):
        if False:
            i = 10
            return i + 15
        self.abortable.push_request()
        try:
            result = self.abortable.apply_async()
            result.abort()
            tid = result.id
            assert self.abortable.is_aborted(task_id=tid)
        finally:
            self.abortable.pop_request()