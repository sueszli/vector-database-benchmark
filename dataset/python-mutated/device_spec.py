"""Class to represent a device."""
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
_VALID_DEVICE_TYPES = frozenset({'CPU', 'GPU', 'TPU', 'CUSTOM', 'EPU'})
_STRING_TO_COMPONENTS_CACHE = {}
_COMPONENTS_TO_STRING_CACHE = {}

def _as_str_or_none(inp):
    if False:
        print('Hello World!')
    return None if inp is None else str(inp)

def _as_int_or_none(inp):
    if False:
        return 10
    return None if inp is None else int(inp)

def _as_device_str_or_none(device_type):
    if False:
        for i in range(10):
            print('nop')
    if device_type in ('cpu', 'gpu'):
        return device_type.upper()
    return _as_str_or_none(device_type)

@tf_export('DeviceSpec', v1=[])
class DeviceSpecV2(object):
    """Represents a (possibly partial) specification for a TensorFlow device.

  `DeviceSpec`s are used throughout TensorFlow to describe where state is stored
  and computations occur. Using `DeviceSpec` allows you to parse device spec
  strings to verify their validity, merge them or compose them programmatically.

  Example:

  ```python
  # Place the operations on device "GPU:0" in the "ps" job.
  device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
  with tf.device(device_spec.to_string()):
    # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
    my_var = tf.Variable(..., name="my_variable")
    squared_var = tf.square(my_var)
  ```

  With eager execution disabled (by default in TensorFlow 1.x and by calling
  disable_eager_execution() in TensorFlow 2.x), the following syntax
  can be used:

  ```python
  tf.compat.v1.disable_eager_execution()

  # Same as previous
  device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
  # No need of .to_string() method.
  with tf.device(device_spec):
    my_var = tf.Variable(..., name="my_variable")
    squared_var = tf.square(my_var)
  ```

  If a `DeviceSpec` is partially specified, it will be merged with other
  `DeviceSpec`s according to the scope in which it is defined. `DeviceSpec`
  components defined in inner scopes take precedence over those defined in
  outer scopes.

  ```python
  gpu0_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
  with tf.device(DeviceSpec(job="train").to_string()):
    with tf.device(gpu0_spec.to_string()):
      # Nodes created here will be assigned to /job:ps/device:GPU:0.
    with tf.device(DeviceSpec(device_type="GPU", device_index=1).to_string()):
      # Nodes created here will be assigned to /job:train/device:GPU:1.
  ```

  A `DeviceSpec` consists of 5 components -- each of
  which is optionally specified:

  * Job: The job name.
  * Replica: The replica index.
  * Task: The task index.
  * Device type: The device type string (e.g. "CPU" or "GPU").
  * Device index: The device index.
  """
    __slots__ = ('_job', '_replica', '_task', '_device_type', '_device_index', '_as_string', '_hash')

    def __init__(self, job=None, replica=None, task=None, device_type=None, device_index=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a new `DeviceSpec` object.\n\n    Args:\n      job: string.  Optional job name.\n      replica: int.  Optional replica index.\n      task: int.  Optional task index.\n      device_type: Optional device type string (e.g. "CPU" or "GPU")\n      device_index: int.  Optional device index.  If left unspecified, device\n        represents \'any\' device_index.\n    '
        self._job = _as_str_or_none(job)
        self._replica = _as_int_or_none(replica)
        self._task = _as_int_or_none(task)
        self._device_type = _as_device_str_or_none(device_type)
        self._device_index = _as_int_or_none(device_index)
        self._as_string = self._components_to_string(job=self._job, replica=self._replica, task=self._task, device_type=self._device_type, device_index=self._device_index)
        self._hash = hash(self.to_string())

    def to_string(self):
        if False:
            print('Hello World!')
        'Return a string representation of this `DeviceSpec`.\n\n    Returns:\n      a string of the form\n      /job:<name>/replica:<id>/task:<id>/device:<device_type>:<id>.\n    '
        return self._as_string

    @classmethod
    def from_string(cls, spec):
        if False:
            return 10
        'Construct a `DeviceSpec` from a string.\n\n    Args:\n      spec: a string of the form\n       /job:<name>/replica:<id>/task:<id>/device:CPU:<id> or\n       /job:<name>/replica:<id>/task:<id>/device:GPU:<id> as cpu and gpu are\n         mutually exclusive. All entries are optional.\n\n    Returns:\n      A DeviceSpec.\n    '
        return cls(*cls._string_to_components(spec))

    def parse_from_string(self, spec):
        if False:
            i = 10
            return i + 15
        'Parse a `DeviceSpec` name into its components.\n\n    **2.x behavior change**:\n\n    In TensorFlow 1.x, this function mutates its own state and returns itself.\n    In 2.x, DeviceSpecs are immutable, and this function will return a\n      DeviceSpec which contains the spec.\n\n    * Recommended:\n\n      ```\n      # my_spec and my_updated_spec are unrelated.\n      my_spec = tf.DeviceSpec.from_string("/CPU:0")\n      my_updated_spec = tf.DeviceSpec.from_string("/GPU:0")\n      with tf.device(my_updated_spec):\n        ...\n      ```\n\n    * Will work in 1.x and 2.x (though deprecated in 2.x):\n\n      ```\n      my_spec = tf.DeviceSpec.from_string("/CPU:0")\n      my_updated_spec = my_spec.parse_from_string("/GPU:0")\n      with tf.device(my_updated_spec):\n        ...\n      ```\n\n    * Will NOT work in 2.x:\n\n      ```\n      my_spec = tf.DeviceSpec.from_string("/CPU:0")\n      my_spec.parse_from_string("/GPU:0")  # <== Will not update my_spec\n      with tf.device(my_spec):\n        ...\n      ```\n\n    In general, `DeviceSpec.from_string` should completely replace\n    `DeviceSpec.parse_from_string`, and `DeviceSpec.replace` should\n    completely replace setting attributes directly.\n\n    Args:\n      spec: an optional string of the form\n       /job:<name>/replica:<id>/task:<id>/device:CPU:<id> or\n       /job:<name>/replica:<id>/task:<id>/device:GPU:<id> as cpu and gpu are\n         mutually exclusive. All entries are optional.\n\n    Returns:\n      The `DeviceSpec`.\n\n    Raises:\n      ValueError: if the spec was not valid.\n    '
        return self.from_string(spec)

    def make_merged_spec(self, dev):
        if False:
            for i in range(10):
                print('nop')
        'Returns a new DeviceSpec which incorporates `dev`.\n\n    When combining specs, `dev` will take precedence over the current spec.\n    So for instance:\n    ```\n    first_spec = tf.DeviceSpec(job=0, device_type="CPU")\n    second_spec = tf.DeviceSpec(device_type="GPU")\n    combined_spec = first_spec.make_merged_spec(second_spec)\n    ```\n\n    is equivalent to:\n    ```\n    combined_spec = tf.DeviceSpec(job=0, device_type="GPU")\n    ```\n\n    Args:\n      dev: a `DeviceSpec`\n\n    Returns:\n      A new `DeviceSpec` which combines `self` and `dev`\n    '
        return self.__class__(*self._get_combined_properties(dev))

    def replace(self, **kwargs):
        if False:
            while True:
                i = 10
        'Convenience method for making a new DeviceSpec by overriding fields.\n\n    For instance:\n    ```\n    my_spec = DeviceSpec=(job="my_job", device="CPU")\n    my_updated_spec = my_spec.replace(device="GPU")\n    my_other_spec = my_spec.replace(device=None)\n    ```\n\n    Args:\n      **kwargs: This method takes the same args as the DeviceSpec constructor\n\n    Returns:\n      A DeviceSpec with the fields specified in kwargs overridden.\n    '
        init_kwargs = dict(job=self.job, replica=self.replica, task=self.task, device_type=self.device_type, device_index=self.device_index)
        init_kwargs.update(kwargs)
        return self.__class__(**init_kwargs)

    @property
    def job(self):
        if False:
            i = 10
            return i + 15
        return self._job

    @property
    def replica(self):
        if False:
            for i in range(10):
                print('nop')
        return self._replica

    @property
    def task(self):
        if False:
            while True:
                i = 10
        return self._task

    @property
    def device_type(self):
        if False:
            print('Hello World!')
        return self._device_type

    @property
    def device_index(self):
        if False:
            while True:
                i = 10
        return self._device_index

    def _get_combined_properties(self, dev):
        if False:
            while True:
                i = 10
        'Combine the current DeviceSpec with another DeviceSpec.\n\n    The combination of DeviceSpecs is will give priority to dev.\n\n    Args:\n      dev: a `DeviceSpec`\n\n    Returns:\n      A tuple of (job, replica, task, device_type, device_index) which\n      represents the combination of self and dev.\n    '
        return (dev.job if dev.job is not None else self.job, dev.replica if dev.replica is not None else self.replica, dev.task if dev.task is not None else self.task, dev.device_type if dev.device_type is not None else self.device_type, dev.device_index if dev.device_index is not None else self.device_index)

    @staticmethod
    def _get_valid_device_types():
        if False:
            while True:
                i = 10
        valid_device_types = set({})
        physical_devices = pywrap_tfe.TF_ListPluggablePhysicalDevices()
        for device in physical_devices:
            valid_device_types.add(device.decode().split(':')[1])
        valid_device_types = valid_device_types | _VALID_DEVICE_TYPES
        return valid_device_types

    @staticmethod
    def _string_to_components(spec=None):
        if False:
            for i in range(10):
                print('nop')
        'Stateless portion of device spec string parsing.\n\n    Args:\n      spec: An optional string specifying a device specification.\n\n    Returns:\n      The parsed components of `spec`. Note that the result of this function\n      must go through attribute setters of DeviceSpec, and should therefore NOT\n      be used directly.\n    '
        cached_result = _STRING_TO_COMPONENTS_CACHE.get(spec)
        if cached_result is not None:
            return cached_result
        raw_spec = spec
        (job, replica, task, device_type, device_index) = (None, None, None, None, None)
        spec = spec or ''
        splits = [x.split(':') for x in spec.split('/')]
        valid_device_types = DeviceSpecV2._get_valid_device_types()
        for y in splits:
            ly = len(y)
            if y:
                if ly == 2 and y[0] == 'job':
                    job = y[1]
                elif ly == 2 and y[0] == 'replica':
                    replica = y[1]
                elif ly == 2 and y[0] == 'task':
                    task = y[1]
                elif (ly == 1 or ly == 2) and y[0].upper() in valid_device_types:
                    if device_type is not None:
                        raise ValueError(f'Multiple device types are not allowed while parsing the device spec: {spec}.')
                    device_type = y[0].upper()
                    if ly == 2 and y[1] != '*':
                        device_index = int(y[1])
                elif ly == 3 and y[0] == 'device':
                    if device_type is not None:
                        raise ValueError(f'Multiple device types are not allowed while parsing the device spec: {spec}.')
                    device_type = y[1]
                    if y[2] != '*':
                        device_index = int(y[2])
                elif ly and y[0] != '':
                    raise ValueError(f"Unknown attribute '{y[0]}' is encountered while parsing the device spec: '{spec}'.")
        output = (job, replica, task, device_type, device_index)
        _STRING_TO_COMPONENTS_CACHE[raw_spec] = output
        return output

    @staticmethod
    def _components_to_string(job, replica, task, device_type, device_index):
        if False:
            return 10
        'Stateless portion of `to_string` (separated to allow caching).'
        key = (job, replica, task, device_type, device_index)
        cached_result = _COMPONENTS_TO_STRING_CACHE.get(key)
        if cached_result is not None:
            return cached_result
        output = []
        if job is not None:
            output.append('/job:' + job)
        if replica is not None:
            output.append('/replica:' + str(replica))
        if task is not None:
            output.append('/task:' + str(task))
        if device_type is not None:
            device_index_string = '*'
            if device_index is not None:
                device_index_string = str(device_index)
            output.append('/device:%s:%s' % (device_type, device_index_string))
        output = ''.join(output)
        _COMPONENTS_TO_STRING_CACHE[key] = output
        return output

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Checks if the `other` DeviceSpec is same as the current instance, eg have\n\n       same value for all the internal fields.\n\n    Args:\n      other: Another DeviceSpec\n\n    Returns:\n      Return `True` if `other` is also a DeviceSpec instance and has same value\n      as the current instance.\n      Return `False` otherwise.\n    '
        return isinstance(other, self.__class__) and self.to_string() == other.to_string()

    def __hash__(self):
        if False:
            return 10
        return self._hash

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<DeviceSpec(job={self.job}, replica={self.replica}, task={self.task}, device_type={self.device_type}, device_index={self.device_index})>'

@tf_export(v1=['DeviceSpec'])
class DeviceSpecV1(DeviceSpecV2):
    __doc__ = DeviceSpecV2.__doc__
    __slots__ = DeviceSpecV2.__slots__

    @DeviceSpecV2.job.setter
    def job(self, job):
        if False:
            for i in range(10):
                print('nop')
        self._job = _as_str_or_none(job)
        (self._as_string, self._hash) = (None, None)

    @DeviceSpecV2.replica.setter
    def replica(self, replica):
        if False:
            i = 10
            return i + 15
        self._replica = _as_int_or_none(replica)
        (self._as_string, self._hash) = (None, None)

    @DeviceSpecV2.task.setter
    def task(self, task):
        if False:
            return 10
        self._task = _as_int_or_none(task)
        (self._as_string, self._hash) = (None, None)

    @DeviceSpecV2.device_type.setter
    def device_type(self, device_type):
        if False:
            for i in range(10):
                print('nop')
        self._device_type = _as_device_str_or_none(device_type)
        (self._as_string, self._hash) = (None, None)

    @DeviceSpecV2.device_index.setter
    def device_index(self, device_index):
        if False:
            i = 10
            return i + 15
        self._device_index = _as_int_or_none(device_index)
        (self._as_string, self._hash) = (None, None)

    def __hash__(self):
        if False:
            while True:
                i = 10
        if self._hash is None:
            self._hash = hash(self.to_string())
        return self._hash

    def to_string(self):
        if False:
            for i in range(10):
                print('nop')
        if self._as_string is None:
            self._as_string = self._components_to_string(job=self.job, replica=self.replica, task=self.task, device_type=self.device_type, device_index=self.device_index)
        return self._as_string

    def parse_from_string(self, spec):
        if False:
            print('Hello World!')
        (self.job, self.replica, self.task, self.device_type, self.device_index) = self._string_to_components(spec)
        return self

    def merge_from(self, dev):
        if False:
            for i in range(10):
                print('nop')
        'Merge the properties of "dev" into this `DeviceSpec`.\n\n    Note: Will be removed in TensorFlow 2.x since DeviceSpecs will become\n          immutable.\n\n    Args:\n      dev: a `DeviceSpec`.\n    '
        (self.job, self.replica, self.task, self.device_type, self.device_index) = self._get_combined_properties(dev)
    to_string.__doc__ = DeviceSpecV2.to_string.__doc__
    parse_from_string.__doc__ = DeviceSpecV2.parse_from_string.__doc__