"""Device-related support functions."""
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops

def canonicalize(d, default=None):
    if False:
        print('Hello World!')
    'Canonicalize device string.\n\n  If d has missing components, the rest would be deduced from the `default`\n  argument or from \'/replica:0/task:0/device:CPU:0\'. For example:\n    If d = \'/cpu:0\', default=\'/job:worker/task:1\', it returns\n      \'/job:worker/replica:0/task:1/device:CPU:0\'.\n    If d = \'/cpu:0\', default=\'/job:worker\', it returns\n      \'/job:worker/replica:0/task:0/device:CPU:0\'.\n    If d = \'/gpu:0\', default=None, it returns\n      \'/replica:0/task:0/device:GPU:0\'.\n\n  Note: This uses "job:localhost" as the default if executing eagerly.\n\n  Args:\n    d: a device string or tf.config.LogicalDevice\n    default: a string for default device if d doesn\'t have all components.\n\n  Returns:\n    a canonicalized device string.\n  '
    if isinstance(d, context.LogicalDevice):
        d = tf_device.DeviceSpec.from_string(d.name)
    else:
        d = tf_device.DeviceSpec.from_string(d)
    assert d.device_type is None or d.device_type == d.device_type.upper(), "Device type '%s' must be all-caps." % (d.device_type,)
    result = tf_device.DeviceSpec(replica=0, task=0, device_type='CPU', device_index=0)
    if ops.executing_eagerly_outside_functions():
        host_cpu = tf_device.DeviceSpec.from_string(config.list_logical_devices('CPU')[0].name)
        if host_cpu.job:
            result = result.make_merged_spec(host_cpu)
        else:
            result = result.replace(job='localhost')
    if default:
        result = result.make_merged_spec(tf_device.DeviceSpec.from_string(default))
    result = result.make_merged_spec(d)
    return result.to_string()

def canonicalize_without_job_and_task(d):
    if False:
        return 10
    'Partially canonicalize device string.\n\n  This returns device string from `d` without including job and task.\n  This is most useful for parameter server strategy where the device strings are\n  generated on the chief, but executed on workers.\n\n   For example:\n    If d = \'/cpu:0\', default=\'/job:worker/task:1\', it returns\n      \'/replica:0/device:CPU:0\'.\n    If d = \'/cpu:0\', default=\'/job:worker\', it returns\n      \'/replica:0/device:CPU:0\'.\n    If d = \'/gpu:0\', default=None, it returns\n      \'/replica:0/device:GPU:0\'.\n\n  Note: This uses "job:localhost" as the default if executing eagerly.\n\n  Args:\n    d: a device string or tf.config.LogicalDevice\n\n  Returns:\n    a partially canonicalized device string.\n  '
    canonicalized_device = canonicalize(d)
    spec = tf_device.DeviceSpec.from_string(canonicalized_device)
    spec = spec.replace(job=None, task=None, replica=0)
    return spec.to_string()

def resolve(d):
    if False:
        return 10
    'Canonicalize `d` with current device as default.'
    return canonicalize(d, default=current())

class _FakeNodeDef(object):
    """A fake NodeDef for _FakeOperation."""
    __slots__ = ['op', 'name']

    def __init__(self):
        if False:
            print('Hello World!')
        self.op = ''
        self.name = ''

class _FakeOperation(object):
    """A fake Operation object to pass to device functions."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.device = ''
        self.type = ''
        self.name = ''
        self.node_def = _FakeNodeDef()

    def _set_device(self, device):
        if False:
            while True:
                i = 10
        self.device = ops._device_string(device)

    def _set_device_from_string(self, device_str):
        if False:
            for i in range(10):
                print('nop')
        self.device = device_str

def current():
    if False:
        print('Hello World!')
    'Return a string (not canonicalized) for the current device.'
    if ops.executing_eagerly_outside_functions():
        d = context.context().device_name
    else:
        op = _FakeOperation()
        ops.get_default_graph()._apply_device_functions(op)
        d = op.device
    return d

def get_host_for_device(device):
    if False:
        while True:
            i = 10
    'Returns the corresponding host device for the given device.'
    spec = tf_device.DeviceSpec.from_string(device)
    return tf_device.DeviceSpec(job=spec.job, replica=spec.replica, task=spec.task, device_type='CPU', device_index=0).to_string()

def local_devices_from_num_gpus(num_gpus):
    if False:
        print('Hello World!')
    'Returns device strings for local GPUs or CPU.'
    return tuple(('/device:GPU:%d' % i for i in range(num_gpus))) or ('/device:CPU:0',)