from typing import ClassVar, Type, Dict, List, Optional, Union, cast
from pyspark.java_gateway import local_connect_and_auth
from pyspark.resource import ResourceInformation
from pyspark.serializers import read_int, write_int, write_with_length, UTF8Deserializer
from pyspark.errors import PySparkRuntimeError

class TaskContext:
    """
    Contextual information about a task which can be read or mutated during
    execution. To access the TaskContext for a running task, use:
    :meth:`TaskContext.get`.

    .. versionadded:: 2.2.0

    Examples
    --------
    >>> from pyspark import TaskContext

    Get a task context instance from :class:`RDD`.

    >>> spark.sparkContext.setLocalProperty("key1", "value")
    >>> taskcontext = spark.sparkContext.parallelize([1]).map(lambda _: TaskContext.get()).first()
    >>> isinstance(taskcontext.attemptNumber(), int)
    True
    >>> isinstance(taskcontext.partitionId(), int)
    True
    >>> isinstance(taskcontext.stageId(), int)
    True
    >>> isinstance(taskcontext.taskAttemptId(), int)
    True
    >>> taskcontext.getLocalProperty("key1")
    'value'
    >>> isinstance(taskcontext.cpus(), int)
    True

    Get a task context instance from a dataframe via Python UDF.

    >>> from pyspark.sql import Row
    >>> from pyspark.sql.functions import udf
    >>> @udf("STRUCT<anum: INT, partid: INT, stageid: INT, taskaid: INT, prop: STRING, cpus: INT>")
    ... def taskcontext_as_row():
    ...    taskcontext = TaskContext.get()
    ...    return Row(
    ...        anum=taskcontext.attemptNumber(),
    ...        partid=taskcontext.partitionId(),
    ...        stageid=taskcontext.stageId(),
    ...        taskaid=taskcontext.taskAttemptId(),
    ...        prop=taskcontext.getLocalProperty("key2"),
    ...        cpus=taskcontext.cpus())
    ...
    >>> spark.sparkContext.setLocalProperty("key2", "value")
    >>> [(anum, partid, stageid, taskaid, prop, cpus)] = (
    ...     spark.range(1).select(taskcontext_as_row()).first()
    ... )
    >>> isinstance(anum, int)
    True
    >>> isinstance(partid, int)
    True
    >>> isinstance(stageid, int)
    True
    >>> isinstance(taskaid, int)
    True
    >>> prop
    'value'
    >>> isinstance(cpus, int)
    True

    Get a task context instance from a dataframe via Pandas UDF.

    >>> import pandas as pd  # doctest: +SKIP
    >>> from pyspark.sql.functions import pandas_udf
    >>> @pandas_udf("STRUCT<"
    ...     "anum: INT, partid: INT, stageid: INT, taskaid: INT, prop: STRING, cpus: INT>")
    ... def taskcontext_as_row(_):
    ...    taskcontext = TaskContext.get()
    ...    return pd.DataFrame({
    ...        "anum": [taskcontext.attemptNumber()],
    ...        "partid": [taskcontext.partitionId()],
    ...        "stageid": [taskcontext.stageId()],
    ...        "taskaid": [taskcontext.taskAttemptId()],
    ...        "prop": [taskcontext.getLocalProperty("key3")],
    ...        "cpus": [taskcontext.cpus()]
    ...    })  # doctest: +SKIP
    ...
    >>> spark.sparkContext.setLocalProperty("key3", "value")  # doctest: +SKIP
    >>> [(anum, partid, stageid, taskaid, prop, cpus)] = (
    ...     spark.range(1).select(taskcontext_as_row("id")).first()
    ... )  # doctest: +SKIP
    >>> isinstance(anum, int)
    True
    >>> isinstance(partid, int)
    True
    >>> isinstance(stageid, int)
    True
    >>> isinstance(taskaid, int)
    True
    >>> prop
    'value'
    >>> isinstance(cpus, int)
    True
    """
    _taskContext: ClassVar[Optional['TaskContext']] = None
    _attemptNumber: Optional[int] = None
    _partitionId: Optional[int] = None
    _stageId: Optional[int] = None
    _taskAttemptId: Optional[int] = None
    _localProperties: Optional[Dict[str, str]] = None
    _cpus: Optional[int] = None
    _resources: Optional[Dict[str, ResourceInformation]] = None

    def __new__(cls: Type['TaskContext']) -> 'TaskContext':
        if False:
            return 10
        '\n        Even if users construct :class:`TaskContext` instead of using get, give them the singleton.\n        '
        taskContext = cls._taskContext
        if taskContext is not None:
            return taskContext
        cls._taskContext = taskContext = object.__new__(cls)
        return taskContext

    @classmethod
    def _getOrCreate(cls: Type['TaskContext']) -> 'TaskContext':
        if False:
            for i in range(10):
                print('nop')
        'Internal function to get or create global :class:`TaskContext`.'
        if cls._taskContext is None:
            cls._taskContext = TaskContext()
        return cls._taskContext

    @classmethod
    def _setTaskContext(cls: Type['TaskContext'], taskContext: 'TaskContext') -> None:
        if False:
            return 10
        cls._taskContext = taskContext

    @classmethod
    def get(cls: Type['TaskContext']) -> Optional['TaskContext']:
        if False:
            while True:
                i = 10
        '\n        Return the currently active :class:`TaskContext`. This can be called inside of\n        user functions to access contextual information about running tasks.\n\n        Returns\n        -------\n        :class:`TaskContext`, optional\n\n        Notes\n        -----\n        Must be called on the worker, not the driver. Returns ``None`` if not initialized.\n        '
        return cls._taskContext

    def stageId(self) -> int:
        if False:
            return 10
        '\n        The ID of the stage that this task belong to.\n\n        Returns\n        -------\n        int\n            current stage id.\n        '
        return cast(int, self._stageId)

    def partitionId(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        The ID of the RDD partition that is computed by this task.\n\n        Returns\n        -------\n        int\n            current partition id.\n        '
        return cast(int, self._partitionId)

    def attemptNumber(self) -> int:
        if False:
            return 10
        '\n        How many times this task has been attempted.  The first task attempt will be assigned\n        attemptNumber = 0, and subsequent attempts will have increasing attempt numbers.\n\n        Returns\n        -------\n        int\n            current attempt number.\n        '
        return cast(int, self._attemptNumber)

    def taskAttemptId(self) -> int:
        if False:
            return 10
        "\n        An ID that is unique to this task attempt (within the same :class:`SparkContext`,\n        no two task attempts will share the same attempt ID).  This is roughly equivalent\n        to Hadoop's `TaskAttemptID`.\n\n        Returns\n        -------\n        int\n            current task attempt id.\n        "
        return cast(int, self._taskAttemptId)

    def getLocalProperty(self, key: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a local property set upstream in the driver, or None if it is missing.\n\n        Parameters\n        ----------\n        key : str\n            the key of the local property to get.\n\n        Returns\n        -------\n        int\n            the value of the local property.\n        '
        return cast(Dict[str, str], self._localProperties).get(key, None)

    def cpus(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        CPUs allocated to the task.\n\n        Returns\n        -------\n        int\n            the number of CPUs.\n        '
        return cast(int, self._cpus)

    def resources(self) -> Dict[str, ResourceInformation]:
        if False:
            print('Hello World!')
        '\n        Resources allocated to the task. The key is the resource name and the value is information\n        about the resource.\n\n        Returns\n        -------\n        dict\n            a dictionary of a string resource name, and :class:`ResourceInformation`.\n        '
        return cast(Dict[str, ResourceInformation], self._resources)
BARRIER_FUNCTION = 1
ALL_GATHER_FUNCTION = 2

def _load_from_socket(port: Optional[Union[str, int]], auth_secret: str, function: int, all_gather_message: Optional[str]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Load data from a given socket, this is a blocking method thus only return when the socket\n    connection has been closed.\n    '
    (sockfile, sock) = local_connect_and_auth(port, auth_secret)
    sock.settimeout(None)
    if function == BARRIER_FUNCTION:
        write_int(function, sockfile)
    elif function == ALL_GATHER_FUNCTION:
        write_int(function, sockfile)
        write_with_length(cast(str, all_gather_message).encode('utf-8'), sockfile)
    else:
        raise ValueError('Unrecognized function type')
    sockfile.flush()
    len = read_int(sockfile)
    res = []
    for i in range(len):
        res.append(UTF8Deserializer().loads(sockfile))
    sockfile.close()
    sock.close()
    return res

class BarrierTaskContext(TaskContext):
    """
    A :class:`TaskContext` with extra contextual info and tooling for tasks in a barrier stage.
    Use :func:`BarrierTaskContext.get` to obtain the barrier context for a running barrier task.

    .. versionadded:: 2.4.0

    Notes
    -----
    This API is experimental

    Examples
    --------
    Set a barrier, and execute it with RDD.

    >>> from pyspark import BarrierTaskContext
    >>> def block_and_do_something(itr):
    ...     taskcontext = BarrierTaskContext.get()
    ...     # Do something.
    ...
    ...     # Wait until all tasks finished.
    ...     taskcontext.barrier()
    ...
    ...     return itr
    ...
    >>> rdd = spark.sparkContext.parallelize([1])
    >>> rdd.barrier().mapPartitions(block_and_do_something).collect()
    [1]
    """
    _port: ClassVar[Optional[Union[str, int]]] = None
    _secret: ClassVar[Optional[str]] = None

    @classmethod
    def _getOrCreate(cls: Type['BarrierTaskContext']) -> 'BarrierTaskContext':
        if False:
            return 10
        '\n        Internal function to get or create global :class:`BarrierTaskContext`. We need to make sure\n        :class:`BarrierTaskContext` is returned from here because it is needed in python worker\n        reuse scenario, see SPARK-25921 for more details.\n        '
        if not isinstance(cls._taskContext, BarrierTaskContext):
            cls._taskContext = object.__new__(cls)
        return cls._taskContext

    @classmethod
    def get(cls: Type['BarrierTaskContext']) -> 'BarrierTaskContext':
        if False:
            return 10
        '\n        Return the currently active :class:`BarrierTaskContext`.\n        This can be called inside of user functions to access contextual information about\n        running tasks.\n\n        Notes\n        -----\n        Must be called on the worker, not the driver. Returns ``None`` if not initialized.\n        An Exception will raise if it is not in a barrier stage.\n\n        This API is experimental\n        '
        if not isinstance(cls._taskContext, BarrierTaskContext):
            raise PySparkRuntimeError(error_class='NOT_IN_BARRIER_STAGE', message_parameters={})
        return cls._taskContext

    @classmethod
    def _initialize(cls: Type['BarrierTaskContext'], port: Optional[Union[str, int]], secret: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize :class:`BarrierTaskContext`, other methods within :class:`BarrierTaskContext`\n        can only be called after BarrierTaskContext is initialized.\n        '
        cls._port = port
        cls._secret = secret

    def barrier(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets a global barrier and waits until all tasks in this stage hit this barrier.\n        Similar to `MPI_Barrier` function in MPI, this function blocks until all tasks\n        in the same stage have reached this routine.\n\n        .. versionadded:: 2.4.0\n\n        Notes\n        -----\n        This API is experimental\n\n        In a barrier stage, each task much have the same number of `barrier()`\n        calls, in all possible code branches. Otherwise, you may get the job hanging\n        or a `SparkException` after timeout.\n        '
        if self._port is None or self._secret is None:
            raise PySparkRuntimeError(error_class='CALL_BEFORE_INITIALIZE', message_parameters={'func_name': 'barrier', 'object': 'BarrierTaskContext'})
        else:
            _load_from_socket(self._port, self._secret, BARRIER_FUNCTION)

    def allGather(self, message: str='') -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function blocks until all tasks in the same stage have reached this routine.\n        Each task passes in a message and returns with a list of all the messages passed in\n        by each of those tasks.\n\n        .. versionadded:: 3.0.0\n\n        Notes\n        -----\n        This API is experimental\n\n        In a barrier stage, each task much have the same number of `barrier()`\n        calls, in all possible code branches. Otherwise, you may get the job hanging\n        or a `SparkException` after timeout.\n        '
        if not isinstance(message, str):
            raise TypeError('Argument `message` must be of type `str`')
        elif self._port is None or self._secret is None:
            raise PySparkRuntimeError(error_class='CALL_BEFORE_INITIALIZE', message_parameters={'func_name': 'allGather', 'object': 'BarrierTaskContext'})
        else:
            return _load_from_socket(self._port, self._secret, ALL_GATHER_FUNCTION, message)

    def getTaskInfos(self) -> List['BarrierTaskInfo']:
        if False:
            i = 10
            return i + 15
        "\n        Returns :class:`BarrierTaskInfo` for all tasks in this barrier stage,\n        ordered by partition ID.\n\n        .. versionadded:: 2.4.0\n\n        Notes\n        -----\n        This API is experimental\n\n        Examples\n        --------\n        >>> from pyspark import BarrierTaskContext\n        >>> rdd = spark.sparkContext.parallelize([1])\n        >>> barrier_info = rdd.barrier().mapPartitions(\n        ...     lambda _: [BarrierTaskContext.get().getTaskInfos()]).collect()[0][0]\n        >>> barrier_info.address\n        '...:...'\n        "
        if self._port is None or self._secret is None:
            raise PySparkRuntimeError(error_class='CALL_BEFORE_INITIALIZE', message_parameters={'func_name': 'getTaskInfos', 'object': 'BarrierTaskContext'})
        else:
            addresses = cast(Dict[str, str], self._localProperties).get('addresses', '')
            return [BarrierTaskInfo(h.strip()) for h in addresses.split(',')]

class BarrierTaskInfo:
    """
    Carries all task infos of a barrier task.

    .. versionadded:: 2.4.0

    Attributes
    ----------
    address : str
        The IPv4 address (host:port) of the executor that the barrier task is running on

    Notes
    -----
    This API is experimental
    """

    def __init__(self, address: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.address = address

def _test() -> None:
    if False:
        for i in range(10):
            print('nop')
    import doctest
    import sys
    from pyspark.sql import SparkSession
    globs = globals().copy()
    globs['spark'] = SparkSession.builder.master('local[2]').appName('taskcontext tests').getOrCreate()
    (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
    globs['spark'].stop()
    if failure_count:
        sys.exit(-1)
if __name__ == '__main__':
    _test()