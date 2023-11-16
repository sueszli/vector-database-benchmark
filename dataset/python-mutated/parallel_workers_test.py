import unittest
from caffe2.python import workspace, core
import caffe2.python.parallel_workers as parallel_workers

def create_queue():
    if False:
        return 10
    queue = 'queue'
    workspace.RunOperatorOnce(core.CreateOperator('CreateBlobsQueue', [], [queue], num_blobs=1, capacity=1000))
    for i in range(100):
        workspace.C.Workspace.current.create_blob('blob_' + str(i))
        workspace.C.Workspace.current.create_blob('status_blob_' + str(i))
    workspace.C.Workspace.current.create_blob('dequeue_blob')
    workspace.C.Workspace.current.create_blob('status_blob')
    return queue

def create_worker(queue, get_blob_data):
    if False:
        i = 10
        return i + 15

    def dummy_worker(worker_id):
        if False:
            while True:
                i = 10
        blob = 'blob_' + str(worker_id)
        workspace.FeedBlob(blob, get_blob_data(worker_id))
        workspace.RunOperatorOnce(core.CreateOperator('SafeEnqueueBlobs', [queue, blob], [blob, 'status_blob_' + str(worker_id)]))
    return dummy_worker

def dequeue_value(queue):
    if False:
        return 10
    dequeue_blob = 'dequeue_blob'
    workspace.RunOperatorOnce(core.CreateOperator('SafeDequeueBlobs', [queue], [dequeue_blob, 'status_blob']))
    return workspace.FetchBlob(dequeue_blob)

class ParallelWorkersTest(unittest.TestCase):

    def testParallelWorkers(self):
        if False:
            print('Hello World!')
        workspace.ResetWorkspace()
        queue = create_queue()
        dummy_worker = create_worker(queue, str)
        worker_coordinator = parallel_workers.init_workers(dummy_worker)
        worker_coordinator.start()
        for _ in range(10):
            value = dequeue_value(queue)
            self.assertTrue(value in [b'0', b'1'], 'Got unexpected value ' + str(value))
        self.assertTrue(worker_coordinator.stop())

    def testParallelWorkersInitFun(self):
        if False:
            print('Hello World!')
        workspace.ResetWorkspace()
        queue = create_queue()
        dummy_worker = create_worker(queue, lambda worker_id: workspace.FetchBlob('data'))
        workspace.FeedBlob('data', 'not initialized')

        def init_fun(worker_coordinator, global_coordinator):
            if False:
                for i in range(10):
                    print('nop')
            workspace.FeedBlob('data', 'initialized')
        worker_coordinator = parallel_workers.init_workers(dummy_worker, init_fun=init_fun)
        worker_coordinator.start()
        for _ in range(10):
            value = dequeue_value(queue)
            self.assertEqual(value, b'initialized', 'Got unexpected value ' + str(value))
        worker_coordinator.stop()

    def testParallelWorkersShutdownFun(self):
        if False:
            while True:
                i = 10
        workspace.ResetWorkspace()
        queue = create_queue()
        dummy_worker = create_worker(queue, str)
        workspace.FeedBlob('data', 'not shutdown')

        def shutdown_fun():
            if False:
                while True:
                    i = 10
            workspace.FeedBlob('data', 'shutdown')
        worker_coordinator = parallel_workers.init_workers(dummy_worker, shutdown_fun=shutdown_fun)
        worker_coordinator.start()
        self.assertTrue(worker_coordinator.stop())
        data = workspace.FetchBlob('data')
        self.assertEqual(data, b'shutdown', 'Got unexpected value ' + str(data))