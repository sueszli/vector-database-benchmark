from multiprocessing import Process, Queue
import numpy as np
from caffe2.python import core, workspace

class StoreOpsTests:

    @classmethod
    def _test_set_get(cls, queue, create_store_handler_fn, index, num_procs):
        if False:
            i = 10
            return i + 15
        store_handler = create_store_handler_fn()
        blob = 'blob'
        value = np.full(1, 1, np.float32)
        if index == num_procs - 1:
            workspace.FeedBlob(blob, value)
            workspace.RunOperatorOnce(core.CreateOperator('StoreSet', [store_handler, blob], [], blob_name=blob))
        output_blob = 'output_blob'
        workspace.RunOperatorOnce(core.CreateOperator('StoreGet', [store_handler], [output_blob], blob_name=blob))
        try:
            np.testing.assert_array_equal(workspace.FetchBlob(output_blob), 1)
        except AssertionError as err:
            queue.put(err)
        workspace.ResetWorkspace()

    @classmethod
    def test_set_get(cls, create_store_handler_fn):
        if False:
            for i in range(10):
                print('nop')
        queue = Queue()
        num_procs = 4
        procs = []
        for index in range(num_procs):
            proc = Process(target=cls._test_set_get, args=(queue, create_store_handler_fn, index, num_procs))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
        if not queue.empty():
            raise queue.get()

    @classmethod
    def test_get_timeout(cls, create_store_handler_fn):
        if False:
            print('Hello World!')
        store_handler = create_store_handler_fn()
        net = core.Net('get_missing_blob')
        net.StoreGet([store_handler], 1, blob_name='blob')
        workspace.RunNetOnce(net)