"""
Salt Util for getting system information with the Performance Data Helper (pdh).
Counter information is gathered from current activity or log files.

Usage:

.. code-block:: python

    import salt.utils.win_pdh

    # Get a list of Counter objects
    salt.utils.win_pdh.list_objects()

    # Get a list of ``Processor`` instances
    salt.utils.win_pdh.list_instances('Processor')

    # Get a list of ``Processor`` counters
    salt.utils.win_pdh.list_counters('Processor')

    # Get the value of a single counter
    # \\Processor(*)\\% Processor Time
    salt.utils.win_pdh.get_counter('Processor', '*', '% Processor Time')

    # Get the values of multiple counters
    counter_list = [('Processor', '*', '% Processor Time'),
                    ('System', None, 'Context Switches/sec'),
                    ('Memory', None, 'Pages/sec'),
                    ('Server Work Queues', '*', 'Queue Length')]
    salt.utils.win_pdh.get_counters(counter_list)

    # Get all counters for the Processor object
    salt.utils.win_pdh.get_all_counters('Processor')
"""
import logging
import time
import salt.utils.platform
from salt.exceptions import CommandExecutionError
try:
    import pywintypes
    import win32pdh
    HAS_WINDOWS_MODULES = True
except ImportError:
    HAS_WINDOWS_MODULES = False
log = logging.getLogger(__file__)
__virtualname__ = 'pdh'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only works on Windows systems with the PyWin32\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'salt.utils.win_pdh: Requires Windows')
    if not HAS_WINDOWS_MODULES:
        return (False, 'salt.utils.win_pdh: Missing required modules')
    return __virtualname__

class Counter:
    """
    Counter object
    Has enumerations and functions for working with counters
    """
    PERF_SIZE_DWORD = 0
    PERF_SIZE_LARGE = 256
    PERF_SIZE_ZERO = 512
    PERF_SIZE_VARIABLE_LEN = 768
    PERF_TYPE_NUMBER = 0
    PERF_TYPE_COUNTER = 1024
    PERF_TYPE_TEXT = 2048
    PERF_TYPE_ZERO = 3072
    PERF_NUMBER_HEX = 0
    PERF_NUMBER_DECIMAL = 65536
    PERF_NUMBER_DEC_1000 = 131072
    PERF_COUNTER_VALUE = 0
    PERF_COUNTER_RATE = 65536
    PERF_COUNTER_FRACTION = 131072
    PERF_COUNTER_BASE = 196608
    PERF_COUNTER_ELAPSED = 262144
    PERF_COUNTER_QUEUE_LEN = 327680
    PERF_COUNTER_HISTOGRAM = 393216
    PERF_TEXT_UNICODE = 0
    PERF_TEXT_ASCII = 65536
    PERF_TIMER_TICK = 0
    PERF_TIMER_100NS = 1048576
    PERF_OBJECT_TIMER = 2097152
    PERF_DELTA_COUNTER = 4194304
    PERF_DELTA_BASE = 8388608
    PERF_INVERSE_COUNTER = 16777216
    PERF_MULTI_COUNTER = 33554432
    PERF_DISPLAY_NO_SUFFIX = 0
    PERF_DISPLAY_PER_SEC = 268435456
    PERF_DISPLAY_PERCENT = 536870912
    PERF_DISPLAY_SECONDS = 805306368
    PERF_DISPLAY_NO_SHOW = 1073741824

    def build_counter(obj, instance, instance_index, counter):
        if False:
            for i in range(10):
                print('nop')
        "\n        Makes a fully resolved counter path. Counter names are formatted like\n        this:\n\n        ``\\Processor(*)\\% Processor Time``\n\n        The above breaks down like this:\n\n            obj = 'Processor'\n            instance = '*'\n            counter = '% Processor Time'\n\n        Args:\n\n            obj (str):\n                The top level object\n\n            instance (str):\n                The instance of the object\n\n            instance_index (int):\n                The index of the instance. Can usually be 0\n\n            counter (str):\n                The name of the counter\n\n        Returns:\n            Counter: A Counter object with the path if valid\n\n        Raises:\n            CommandExecutionError: If the path is invalid\n        "
        path = win32pdh.MakeCounterPath((None, obj, instance, None, instance_index, counter), 0)
        if win32pdh.ValidatePath(path) == 0:
            return Counter(path, obj, instance, instance_index, counter)
        raise CommandExecutionError('Invalid counter specified: {}'.format(path))
    build_counter = staticmethod(build_counter)

    def __init__(self, path, obj, instance, index, counter):
        if False:
            return 10
        self.path = path
        self.obj = obj
        self.instance = instance
        self.index = index
        self.counter = counter
        self.handle = None
        self.info = None
        self.type = None

    def add_to_query(self, query):
        if False:
            while True:
                i = 10
        '\n        Add the current path to the query\n\n        Args:\n            query (obj):\n                The handle to the query to add the counter\n        '
        self.handle = win32pdh.AddCounter(query, self.path)

    def get_info(self):
        if False:
            print('Hello World!')
        '\n        Get information about the counter\n\n        .. note::\n            GetCounterInfo sometimes crashes in the wrapper code. Fewer crashes\n            if this is called after sampling data.\n        '
        if not self.info:
            ci = win32pdh.GetCounterInfo(self.handle, 0)
            self.info = {'type': ci[0], 'version': ci[1], 'scale': ci[2], 'default_scale': ci[3], 'user_data': ci[4], 'query_user_data': ci[5], 'full_path': ci[6], 'machine_name': ci[7][0], 'object_name': ci[7][1], 'instance_name': ci[7][2], 'parent_instance': ci[7][3], 'instance_index': ci[7][4], 'counter_name': ci[7][5], 'explain_text': ci[8]}
        return self.info

    def value(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the counter value\n\n        Returns:\n            long: The counter value\n        '
        (counter_type, value) = win32pdh.GetFormattedCounterValue(self.handle, win32pdh.PDH_FMT_DOUBLE)
        self.type = counter_type
        return value

    def type_string(self):
        if False:
            return 10
        '\n        Returns the names of the flags that are set in the Type field\n\n        It can be used to format the counter.\n        '
        type = self.get_info()['type']
        type_list = []
        for member in dir(self):
            if member.startswith('PERF_'):
                bit = getattr(self, member)
                if bit and bit & type:
                    type_list.append(member[5:])
        return type_list

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.path

def list_objects():
    if False:
        print('Hello World!')
    '\n    Get a list of available counter objects on the system\n\n    Returns:\n        list: A list of counter objects\n    '
    return sorted(win32pdh.EnumObjects(None, None, -1, 0))

def list_counters(obj):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a list of counters available for the object\n\n    Args:\n        obj (str):\n            The name of the counter object. You can get a list of valid names\n            using the ``list_objects`` function\n\n    Returns:\n        list: A list of counters available to the passed object\n    '
    return win32pdh.EnumObjectItems(None, None, obj, -1, 0)[0]

def list_instances(obj):
    if False:
        return 10
    '\n    Get a list of instances available for the object\n\n    Args:\n        obj (str):\n            The name of the counter object. You can get a list of valid names\n            using the ``list_objects`` function\n\n    Returns:\n        list: A list of instances available to the passed object\n    '
    return win32pdh.EnumObjectItems(None, None, obj, -1, 0)[1]

def build_counter_list(counter_list):
    if False:
        print('Hello World!')
    "\n    Create a list of Counter objects to be used in the pdh query\n\n    Args:\n        counter_list (list):\n            A list of tuples containing counter information. Each tuple should\n            contain the object, instance, and counter name. For example, to\n            get the ``% Processor Time`` counter for all Processors on the\n            system (``\\Processor(*)\\% Processor Time``) you would pass a tuple\n            like this:\n\n            ```\n            counter_list = [('Processor', '*', '% Processor Time')]\n            ```\n\n            If there is no ``instance`` for the counter, pass ``None``\n\n            Multiple counters can be passed like so:\n\n            ```\n            counter_list = [('Processor', '*', '% Processor Time'),\n                            ('System', None, 'Context Switches/sec')]\n            ```\n\n            .. note::\n                Invalid counters are ignored\n\n    Returns:\n        list: A list of Counter objects\n    "
    counters = []
    index = 0
    for (obj, instance, counter_name) in counter_list:
        try:
            counter = Counter.build_counter(obj, instance, index, counter_name)
            index += 1
            counters.append(counter)
        except CommandExecutionError as exc:
            log.debug(exc.strerror)
            continue
    return counters

def get_all_counters(obj, instance_list=None):
    if False:
        print('Hello World!')
    '\n    Get the values for all counters available to a Counter object\n\n    Args:\n\n        obj (str):\n            The name of the counter object. You can get a list of valid names\n            using the ``list_objects`` function\n\n        instance_list (list):\n            A list of instances to return. Use this to narrow down the counters\n            that are returned.\n\n            .. note::\n                ``_Total`` is returned as ``*``\n    '
    (counters, instances_avail) = win32pdh.EnumObjectItems(None, None, obj, -1, 0)
    if instance_list is None:
        instance_list = instances_avail
    if not isinstance(instance_list, list):
        instance_list = [instance_list]
    counter_list = []
    for counter in counters:
        for instance in instance_list:
            instance = '*' if instance.lower() == '_total' else instance
            counter_list.append((obj, instance, counter))
        else:
            counter_list.append((obj, None, counter))
    return get_counters(counter_list) if counter_list else {}

def get_counters(counter_list):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the values for the passes list of counters\n\n    Args:\n        counter_list (list):\n            A list of counters to lookup\n\n    Returns:\n        dict: A dictionary of counters and their values\n    '
    if not isinstance(counter_list, list):
        raise CommandExecutionError('counter_list must be a list of tuples')
    try:
        query = win32pdh.OpenQuery()
        counters = build_counter_list(counter_list)
        for counter in counters:
            counter.add_to_query(query)
        win32pdh.CollectQueryData(query)
        time.sleep(1)
        win32pdh.CollectQueryData(query)
        ret = {}
        for counter in counters:
            try:
                ret.update({counter.path: counter.value()})
            except pywintypes.error as exc:
                if exc.strerror == 'No data to return.':
                    continue
                else:
                    raise
    except pywintypes.error as exc:
        if exc.strerror == 'No data to return.':
            return {}
        else:
            raise
    finally:
        win32pdh.CloseQuery(query)
    return ret

def get_counter(obj, instance, counter):
    if False:
        print('Hello World!')
    '\n    Get the value of a single counter\n\n    Args:\n\n        obj (str):\n            The name of the counter object. You can get a list of valid names\n            using the ``list_objects`` function\n\n        instance (str):\n            The counter instance you wish to return. Get a list of instances\n            using the ``list_instances`` function\n\n            .. note::\n                ``_Total`` is returned as ``*``\n\n        counter (str):\n            The name of the counter. Get a list of counters using the\n            ``list_counters`` function\n    '
    return get_counters([(obj, instance, counter)])