from caffe2.python import core, dataio
from caffe2.python.task import TaskGroup
import logging
logger = logging.getLogger(__name__)

class _QueueReader(dataio.Reader):

    def __init__(self, wrapper, num_dequeue_records=1):
        if False:
            i = 10
            return i + 15
        assert wrapper.schema is not None, 'Queue needs a schema in order to be read from.'
        dataio.Reader.__init__(self, wrapper.schema())
        self._wrapper = wrapper
        self._num_dequeue_records = num_dequeue_records

    def setup_ex(self, init_net, exit_net):
        if False:
            i = 10
            return i + 15
        exit_net.CloseBlobsQueue([self._wrapper.queue()], 0)

    def read_ex(self, local_init_net, local_finish_net):
        if False:
            print('Hello World!')
        self._wrapper._new_reader(local_init_net)
        dequeue_net = core.Net('dequeue')
        (fields, status_blob) = dequeue(dequeue_net, self._wrapper.queue(), len(self.schema().field_names()), field_names=self.schema().field_names(), num_records=self._num_dequeue_records)
        return ([dequeue_net], status_blob, fields)

    def read(self, net):
        if False:
            while True:
                i = 10
        (net, _, fields) = self.read_ex(net, None)
        return (net, fields)

class _QueueWriter(dataio.Writer):

    def __init__(self, wrapper):
        if False:
            print('Hello World!')
        self._wrapper = wrapper

    def setup_ex(self, init_net, exit_net):
        if False:
            print('Hello World!')
        exit_net.CloseBlobsQueue([self._wrapper.queue()], 0)

    def write_ex(self, fields, local_init_net, local_finish_net, status):
        if False:
            while True:
                i = 10
        self._wrapper._new_writer(self.schema(), local_init_net)
        enqueue_net = core.Net('enqueue')
        enqueue(enqueue_net, self._wrapper.queue(), fields, status)
        return [enqueue_net]

class QueueWrapper(dataio.Pipe):

    def __init__(self, handler, schema=None, num_dequeue_records=1):
        if False:
            while True:
                i = 10
        dataio.Pipe.__init__(self, schema, TaskGroup.LOCAL_SETUP)
        self._queue = handler
        self._num_dequeue_records = num_dequeue_records

    def reader(self):
        if False:
            for i in range(10):
                print('nop')
        return _QueueReader(self, num_dequeue_records=self._num_dequeue_records)

    def writer(self):
        if False:
            i = 10
            return i + 15
        return _QueueWriter(self)

    def queue(self):
        if False:
            return 10
        return self._queue

class Queue(QueueWrapper):

    def __init__(self, capacity, schema=None, name='queue', num_dequeue_records=1):
        if False:
            for i in range(10):
                print('nop')
        net = core.Net(name)
        queue_blob = net.AddExternalInput(net.NextName('handler'))
        QueueWrapper.__init__(self, queue_blob, schema, num_dequeue_records=num_dequeue_records)
        self.capacity = capacity
        self._setup_done = False

    def setup(self, global_init_net):
        if False:
            print('Hello World!')
        assert self._schema, 'This queue does not have a schema.'
        self._setup_done = True
        global_init_net.CreateBlobsQueue([], [self._queue], capacity=self.capacity, num_blobs=len(self._schema.field_names()), field_names=self._schema.field_names())

def enqueue(net, queue, data_blobs, status=None):
    if False:
        print('Hello World!')
    if status is None:
        status = net.NextName('status')
    queue_blobs = []
    for blob in data_blobs:
        if blob not in queue_blobs:
            queue_blobs.append(blob)
        else:
            logger.warning('Need to copy blob {} to enqueue'.format(blob))
            queue_blobs.append(net.Copy(blob))
    results = net.SafeEnqueueBlobs([queue] + queue_blobs, queue_blobs + [status])
    return results[-1]

def dequeue(net, queue, num_blobs, status=None, field_names=None, num_records=1):
    if False:
        print('Hello World!')
    if field_names is not None:
        assert len(field_names) == num_blobs
        data_names = [net.NextName(name) for name in field_names]
    else:
        data_names = [net.NextName('data', i) for i in range(num_blobs)]
    if status is None:
        status = net.NextName('status')
    results = net.SafeDequeueBlobs(queue, data_names + [status], num_records=num_records)
    results = list(results)
    status_blob = results.pop(-1)
    return (results, status_blob)

def close_queue(step, *queues):
    if False:
        i = 10
        return i + 15
    close_net = core.Net('close_queue_net')
    for queue in queues:
        close_net.CloseBlobsQueue([queue], 0)
    close_step = core.execution_step('%s_step' % str(close_net), close_net)
    return core.execution_step('%s_wraper_step' % str(close_net), [step, close_step])