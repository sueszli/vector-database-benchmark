import typing
import uuid
from enum import Enum
from typing import Callable, Union, List, cast, Optional, overload
from pyflink.util.java_utils import get_j_env_configuration
from pyflink.common import typeinfo, ExecutionConfig, Row
from pyflink.common.typeinfo import RowTypeInfo, Types, TypeInformation, _from_java_type
from pyflink.common.watermark_strategy import WatermarkStrategy, TimestampAssigner
from pyflink.datastream.connectors import Sink
from pyflink.datastream.functions import _get_python_env, FlatMapFunction, MapFunction, Function, FunctionWrapper, SinkFunction, FilterFunction, KeySelector, ReduceFunction, CoMapFunction, CoFlatMapFunction, Partitioner, RuntimeContext, ProcessFunction, KeyedProcessFunction, KeyedCoProcessFunction, WindowFunction, ProcessWindowFunction, InternalWindowFunction, InternalIterableWindowFunction, InternalIterableProcessWindowFunction, CoProcessFunction, InternalSingleValueWindowFunction, InternalSingleValueProcessWindowFunction, PassThroughWindowFunction, AggregateFunction, NullByteKeySelector, AllWindowFunction, InternalIterableAllWindowFunction, ProcessAllWindowFunction, InternalIterableProcessAllWindowFunction, BroadcastProcessFunction, KeyedBroadcastProcessFunction, InternalSingleValueAllWindowFunction, PassThroughAllWindowFunction, InternalSingleValueProcessAllWindowFunction
from pyflink.datastream.output_tag import OutputTag
from pyflink.datastream.slot_sharing_group import SlotSharingGroup
from pyflink.datastream.state import ListStateDescriptor, StateDescriptor, ReducingStateDescriptor, AggregatingStateDescriptor, MapStateDescriptor, ReducingState
from pyflink.datastream.utils import convert_to_python_obj
from pyflink.datastream.window import CountTumblingWindowAssigner, CountSlidingWindowAssigner, CountWindowSerializer, TimeWindowSerializer, Trigger, WindowAssigner, WindowOperationDescriptor, GlobalWindowSerializer, MergingWindowAssigner
from pyflink.java_gateway import get_gateway
from pyflink.util.java_utils import to_jarray
__all__ = ['CloseableIterator', 'DataStream', 'KeyedStream', 'ConnectedStreams', 'WindowedStream', 'DataStreamSink', 'CloseableIterator', 'BroadcastStream', 'BroadcastConnectedStream']
WINDOW_STATE_NAME = 'window-contents'

class DataStream(object):
    """
    A DataStream represents a stream of elements of the same type. A DataStream can be transformed
    into another DataStream by applying a transformation as for example:

    ::
        >>> DataStream.map(MapFunctionImpl())
        >>> DataStream.filter(FilterFunctionImpl())
    """

    def __init__(self, j_data_stream):
        if False:
            return 10
        self._j_data_stream = j_data_stream

    def get_name(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Gets the name of the current data stream. This name is used by the visualization and logging\n        during runtime.\n\n        :return: Name of the stream.\n        '
        return self._j_data_stream.getName()

    def name(self, name: str) -> 'DataStream':
        if False:
            while True:
                i = 10
        '\n        Sets the name of the current data stream. This name is used by the visualization and logging\n        during runtime.\n\n        :param name: Name of the stream.\n        :return: The named operator.\n        '
        self._j_data_stream.name(name)
        return self

    def uid(self, uid: str) -> 'DataStream':
        if False:
            return 10
        '\n        Sets an ID for this operator. The specified ID is used to assign the same operator ID across\n        job submissions (for example when starting a job from a savepoint).\n\n        Important: this ID needs to be unique per transformation and job. Otherwise, job submission\n        will fail.\n\n        :param uid: The unique user-specified ID of this transformation.\n        :return: The operator with the specified ID.\n        '
        self._j_data_stream.uid(uid)
        return self

    def set_uid_hash(self, uid_hash: str) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets an user provided hash for this operator. This will be used AS IS the create the\n        JobVertexID. The user provided hash is an alternative to the generated hashed, that is\n        considered when identifying an operator through the default hash mechanics fails (e.g.\n        because of changes between Flink versions).\n\n        Important: this should be used as a workaround or for trouble shooting. The provided hash\n        needs to be unique per transformation and job. Otherwise, job submission will fail.\n        Furthermore, you cannot assign user-specified hash to intermediate nodes in an operator\n        chain and trying so will let your job fail.\n\n        A use case for this is in migration between Flink versions or changing the jobs in a way\n        that changes the automatically generated hashes. In this case, providing the previous hashes\n        directly through this method (e.g. obtained from old logs) can help to reestablish a lost\n        mapping from states to their target operator.\n\n        :param uid_hash: The user provided hash for this operator. This will become the jobVertexID,\n                         which is shown in the logs and web ui.\n        :return: The operator with the user provided hash.\n        '
        self._j_data_stream.setUidHash(uid_hash)
        return self

    def set_parallelism(self, parallelism: int) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        '\n        Sets the parallelism for this operator.\n\n        :param parallelism: THe parallelism for this operator.\n        :return: The operator with set parallelism.\n        '
        self._j_data_stream.setParallelism(parallelism)
        return self

    def set_max_parallelism(self, max_parallelism: int) -> 'DataStream':
        if False:
            return 10
        '\n        Sets the maximum parallelism of this operator.\n\n        The maximum parallelism specifies the upper bound for dynamic scaling. It also defines the\n        number of key groups used for partitioned state.\n\n        :param max_parallelism: Maximum parallelism.\n        :return: The operator with set maximum parallelism.\n        '
        self._j_data_stream.setMaxParallelism(max_parallelism)
        return self

    def get_type(self) -> TypeInformation:
        if False:
            while True:
                i = 10
        '\n        Gets the type of the stream.\n\n        :return: The type of the DataStream.\n        '
        return typeinfo._from_java_type(self._j_data_stream.getType())

    def get_execution_environment(self):
        if False:
            print('Hello World!')
        '\n        Returns the StreamExecutionEnvironment that was used to create this DataStream.\n\n        :return: The Execution Environment.\n        '
        from pyflink.datastream import StreamExecutionEnvironment
        return StreamExecutionEnvironment(j_stream_execution_environment=self._j_data_stream.getExecutionEnvironment())

    def get_execution_config(self) -> ExecutionConfig:
        if False:
            return 10
        return ExecutionConfig(j_execution_config=self._j_data_stream.getExecutionConfig())

    def force_non_parallel(self) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        '\n        Sets the parallelism and maximum parallelism of this operator to one. And mark this operator\n        cannot set a non-1 degree of parallelism.\n\n        :return: The operator with only one parallelism.\n        '
        self._j_data_stream.forceNonParallel()
        return self

    def set_buffer_timeout(self, timeout_millis: int) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        "\n        Sets the buffering timeout for data produced by this operation. The timeout defines how long\n        data may linger ina partially full buffer before being sent over the network.\n\n        Lower timeouts lead to lower tail latencies, but may affect throughput. Timeouts of 1 ms\n        still sustain high throughput, even for jobs with high parallelism.\n\n        A value of '-1' means that the default buffer timeout should be used. A value of '0'\n        indicates that no buffering should happen, and all records/events should be immediately sent\n        through the network, without additional buffering.\n\n        :param timeout_millis: The maximum time between two output flushes.\n        :return: The operator with buffer timeout set.\n        "
        self._j_data_stream.setBufferTimeout(timeout_millis)
        return self

    def start_new_chain(self) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Starts a new task chain beginning at this operator. This operator will be chained (thread\n        co-located for increased performance) to any previous tasks even if possible.\n\n        :return: The operator with chaining set.\n        '
        self._j_data_stream.startNewChain()
        return self

    def disable_chaining(self) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Turns off chaining for this operator so thread co-location will not be used as an\n        optimization.\n        Chaining can be turned off for the whole job by\n        StreamExecutionEnvironment.disableOperatorChaining() however it is not advised for\n        performance consideration.\n\n        :return: The operator with chaining disabled.\n        '
        self._j_data_stream.disableChaining()
        return self

    def slot_sharing_group(self, slot_sharing_group: Union[str, SlotSharingGroup]) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        "\n        Sets the slot sharing group of this operation. Parallel instances of operations that are in\n        the same slot sharing group will be co-located in the same TaskManager slot, if possible.\n\n        Operations inherit the slot sharing group of input operations if all input operations are in\n        the same slot sharing group and no slot sharing group was explicitly specified.\n\n        Initially an operation is in the default slot sharing group. An operation can be put into\n        the default group explicitly by setting the slot sharing group to 'default'.\n\n        :param slot_sharing_group: The slot sharing group name or which contains name and its\n                        resource spec.\n        :return: This operator.\n        "
        if isinstance(slot_sharing_group, SlotSharingGroup):
            self._j_data_stream.slotSharingGroup(slot_sharing_group.get_java_slot_sharing_group())
        else:
            self._j_data_stream.slotSharingGroup(slot_sharing_group)
        return self

    def set_description(self, description: str) -> 'DataStream':
        if False:
            return 10
        '\n        Sets the description for this operator.\n\n        Description is used in json plan and web ui, but not in logging and metrics where only\n        name is available. Description is expected to provide detailed information about the\n        operator, while name is expected to be more simple, providing summary information only,\n        so that we can have more user-friendly logging messages and metric tags without losing\n        useful messages for debugging.\n\n        :param description: The description for this operator.\n        :return: The operator with new description.\n\n        .. versionadded:: 1.15.0\n        '
        self._j_data_stream.setDescription(description)
        return self

    def map(self, func: Union[Callable, MapFunction], output_type: TypeInformation=None) -> 'DataStream':
        if False:
            while True:
                i = 10
        '\n        Applies a Map transformation on a DataStream. The transformation calls a MapFunction for\n        each element of the DataStream. Each MapFunction call returns exactly one element.\n\n        Note that If user does not specify the output data type, the output data will be serialized\n        as pickle primitive byte array.\n\n        :param func: The MapFunction that is called for each element of the DataStream.\n        :param output_type: The type information of the MapFunction output data.\n        :return: The transformed DataStream.\n        '
        if not isinstance(func, MapFunction) and (not callable(func)):
            raise TypeError('The input must be a MapFunction or a callable function')

        class MapProcessFunctionAdapter(ProcessFunction):

            def __init__(self, map_func):
                if False:
                    print('Hello World!')
                if isinstance(map_func, MapFunction):
                    self._open_func = map_func.open
                    self._close_func = map_func.close
                    self._map_func = map_func.map
                else:
                    self._open_func = None
                    self._close_func = None
                    self._map_func = map_func

            def open(self, runtime_context: RuntimeContext):
                if False:
                    i = 10
                    return i + 15
                if self._open_func:
                    self._open_func(runtime_context)

            def close(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self._close_func:
                    self._close_func()

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    return 10
                yield self._map_func(value)
        return self.process(MapProcessFunctionAdapter(func), output_type).name('Map')

    def flat_map(self, func: Union[Callable, FlatMapFunction], output_type: TypeInformation=None) -> 'DataStream':
        if False:
            return 10
        '\n        Applies a FlatMap transformation on a DataStream. The transformation calls a FlatMapFunction\n        for each element of the DataStream. Each FlatMapFunction call can return any number of\n        elements including none.\n\n        :param func: The FlatMapFunction that is called for each element of the DataStream.\n        :param output_type: The type information of output data.\n        :return: The transformed DataStream.\n        '
        if not isinstance(func, FlatMapFunction) and (not callable(func)):
            raise TypeError('The input must be a FlatMapFunction or a callable function')

        class FlatMapProcessFunctionAdapter(ProcessFunction):

            def __init__(self, flat_map_func):
                if False:
                    while True:
                        i = 10
                if isinstance(flat_map_func, FlatMapFunction):
                    self._open_func = flat_map_func.open
                    self._close_func = flat_map_func.close
                    self._flat_map_func = flat_map_func.flat_map
                else:
                    self._open_func = None
                    self._close_func = None
                    self._flat_map_func = flat_map_func

            def open(self, runtime_context: RuntimeContext):
                if False:
                    i = 10
                    return i + 15
                if self._open_func:
                    self._open_func(runtime_context)

            def close(self):
                if False:
                    i = 10
                    return i + 15
                if self._close_func:
                    self._close_func()

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    i = 10
                    return i + 15
                yield from self._flat_map_func(value)
        return self.process(FlatMapProcessFunctionAdapter(func), output_type).name('FlatMap')

    def key_by(self, key_selector: Union[Callable, KeySelector], key_type: TypeInformation=None) -> 'KeyedStream':
        if False:
            i = 10
            return i + 15
        '\n        Creates a new KeyedStream that uses the provided key for partitioning its operator states.\n\n        :param key_selector: The KeySelector to be used for extracting the key for partitioning.\n        :param key_type: The type information describing the key type.\n        :return: The DataStream with partitioned state(i.e. KeyedStream).\n        '
        if not isinstance(key_selector, KeySelector) and (not callable(key_selector)):
            raise TypeError('Parameter key_selector should be type of KeySelector or a callable function.')

        class AddKey(ProcessFunction):

            def __init__(self, key_selector):
                if False:
                    print('Hello World!')
                if isinstance(key_selector, KeySelector):
                    self._key_selector_open_func = key_selector.open
                    self._key_selector_close_func = key_selector.close
                    self._get_key_func = key_selector.get_key
                else:
                    self._key_selector_open_func = None
                    self._key_selector_close_func = None
                    self._get_key_func = key_selector

            def open(self, runtime_context: RuntimeContext):
                if False:
                    while True:
                        i = 10
                if self._key_selector_open_func:
                    self._key_selector_open_func(runtime_context)

            def close(self):
                if False:
                    while True:
                        i = 10
                if self._key_selector_close_func:
                    self._key_selector_close_func()

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    for i in range(10):
                        print('nop')
                yield Row(self._get_key_func(value), value)
        output_type_info = typeinfo._from_java_type(self._j_data_stream.getTransformation().getOutputType())
        if key_type is None:
            key_type = Types.PICKLED_BYTE_ARRAY()
        gateway = get_gateway()
        stream_with_key_info = self.process(AddKey(key_selector), output_type=Types.ROW([key_type, output_type_info]))
        stream_with_key_info.name(gateway.jvm.org.apache.flink.python.util.PythonConfigUtil.STREAM_KEY_BY_MAP_OPERATOR_NAME)
        JKeyByKeySelector = gateway.jvm.KeyByKeySelector
        key_stream = KeyedStream(stream_with_key_info._j_data_stream.keyBy(JKeyByKeySelector(), Types.ROW([key_type]).get_java_type_info()), output_type_info, self)
        return key_stream

    def filter(self, func: Union[Callable, FilterFunction]) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        '\n        Applies a Filter transformation on a DataStream. The transformation calls a FilterFunction\n        for each element of the DataStream and retains only those element for which the function\n        returns true. Elements for which the function returns false are filtered.\n\n        :param func: The FilterFunction that is called for each element of the DataStream.\n        :return: The filtered DataStream.\n        '
        if not isinstance(func, FilterFunction) and (not callable(func)):
            raise TypeError('The input must be a FilterFunction or a callable function')

        class FilterProcessFunctionAdapter(ProcessFunction):

            def __init__(self, filter_func):
                if False:
                    while True:
                        i = 10
                if isinstance(filter_func, FilterFunction):
                    self._open_func = filter_func.open
                    self._close_func = filter_func.close
                    self._filter_func = filter_func.filter
                else:
                    self._open_func = None
                    self._close_func = None
                    self._filter_func = filter_func

            def open(self, runtime_context: RuntimeContext):
                if False:
                    for i in range(10):
                        print('nop')
                if self._open_func:
                    self._open_func(runtime_context)

            def close(self):
                if False:
                    i = 10
                    return i + 15
                if self._close_func:
                    self._close_func()

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    while True:
                        i = 10
                if self._filter_func(value):
                    yield value
        output_type = typeinfo._from_java_type(self._j_data_stream.getTransformation().getOutputType())
        return self.process(FilterProcessFunctionAdapter(func), output_type=output_type).name('Filter')

    def window_all(self, window_assigner: WindowAssigner) -> 'AllWindowedStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Windows this data stream to a AllWindowedStream, which evaluates windows over a non key\n        grouped stream. Elements are put into windows by a WindowAssigner. The grouping of\n        elements is done by window.\n\n        A Trigger can be defined to specify when windows are evaluated. However, WindowAssigners\n        have a default Trigger that is used if a Trigger is not specified.\n\n        :param window_assigner: The WindowAssigner that assigns elements to windows.\n        :return: The trigger windows data stream.\n\n        .. versionadded:: 1.16.0\n        '
        return AllWindowedStream(self, window_assigner)

    def union(self, *streams: 'DataStream') -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Creates a new DataStream by merging DataStream outputs of the same type with each other. The\n        DataStreams merged using this operator will be transformed simultaneously.\n\n        :param streams: The DataStream to union outputwith.\n        :return: The DataStream.\n        '
        j_data_streams = []
        for data_stream in streams:
            if isinstance(data_stream, KeyedStream):
                j_data_streams.append(data_stream._values()._j_data_stream)
            else:
                j_data_streams.append(data_stream._j_data_stream)
        gateway = get_gateway()
        JDataStream = gateway.jvm.org.apache.flink.streaming.api.datastream.DataStream
        j_data_stream_arr = get_gateway().new_array(JDataStream, len(j_data_streams))
        for i in range(len(j_data_streams)):
            j_data_stream_arr[i] = j_data_streams[i]
        j_united_stream = self._j_data_stream.union(j_data_stream_arr)
        return DataStream(j_data_stream=j_united_stream)

    @overload
    def connect(self, ds: 'DataStream') -> 'ConnectedStreams':
        if False:
            while True:
                i = 10
        pass

    @overload
    def connect(self, ds: 'BroadcastStream') -> 'BroadcastConnectedStream':
        if False:
            for i in range(10):
                print('nop')
        pass

    def connect(self, ds: Union['DataStream', 'BroadcastStream']) -> Union['ConnectedStreams', 'BroadcastConnectedStream']:
        if False:
            print('Hello World!')
        '\n        If ds is a :class:`DataStream`, creates a new :class:`ConnectedStreams` by connecting\n        DataStream outputs of (possible) different types with each other. The DataStreams connected\n        using this operator can be used with CoFunctions to apply joint transformations.\n\n        If ds is a :class:`BroadcastStream`, creates a new :class:`BroadcastConnectedStream` by\n        connecting the current :class:`DataStream` with a :class:`BroadcastStream`. The latter can\n        be created using the :meth:`broadcast` method. The resulting stream can be further processed\n        using the :meth:`BroadcastConnectedStream.process` method.\n\n        :param ds: The DataStream or BroadcastStream with which this stream will be connected.\n        :return: The ConnectedStreams or BroadcastConnectedStream.\n\n        .. versionchanged:: 1.16.0\n           Support connect BroadcastStream\n        '
        if isinstance(ds, BroadcastStream):
            return BroadcastConnectedStream(self, ds, cast(BroadcastStream, ds).broadcast_state_descriptors)
        return ConnectedStreams(self, ds)

    def shuffle(self) -> 'DataStream':
        if False:
            return 10
        '\n        Sets the partitioning of the DataStream so that the output elements are shuffled uniformly\n        randomly to the next operation.\n\n        :return: The DataStream with shuffle partitioning set.\n        '
        return DataStream(self._j_data_stream.shuffle())

    def project(self, *field_indexes: int) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Initiates a Project transformation on a Tuple DataStream.\n\n        Note that only Tuple DataStreams can be projected.\n\n        :param field_indexes: The field indexes of the input tuples that are retained. The order of\n                              fields in the output tuple corresponds to the order of field indexes.\n        :return: The projected DataStream.\n        '
        if not isinstance(self.get_type(), typeinfo.TupleTypeInfo):
            raise Exception('Only Tuple DataStreams can be projected.')
        gateway = get_gateway()
        j_index_arr = gateway.new_array(gateway.jvm.int, len(field_indexes))
        for i in range(len(field_indexes)):
            j_index_arr[i] = field_indexes[i]
        return DataStream(self._j_data_stream.project(j_index_arr))

    def rescale(self) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the partitioning of the DataStream so that the output elements are distributed evenly\n        to a subset of instances of the next operation in a round-robin fashion.\n\n        The subset of downstream operations to which the upstream operation sends elements depends\n        on the degree of parallelism of both the upstream and downstream operation. For example, if\n        the upstream operation has parallelism 2 and the downstream operation has parallelism 4,\n        then one upstream operation would distribute elements to two downstream operations. If, on\n        the other hand, the downstream operation has parallelism 4 then two upstream operations will\n        distribute to one downstream operation while the other two upstream operations will\n        distribute to the other downstream operations.\n\n        In cases where the different parallelisms are not multiples of each one or several\n        downstream operations will have a differing number of inputs from upstream operations.\n\n        :return: The DataStream with rescale partitioning set.\n        '
        return DataStream(self._j_data_stream.rescale())

    def rebalance(self) -> 'DataStream':
        if False:
            return 10
        '\n        Sets the partitioning of the DataStream so that the output elements are distributed evenly\n        to instances of the next operation in a round-robin fashion.\n\n        :return: The DataStream with rebalance partition set.\n        '
        return DataStream(self._j_data_stream.rebalance())

    def forward(self) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the partitioning of the DataStream so that the output elements are forwarded to the\n        local sub-task of the next operation.\n\n        :return: The DataStream with forward partitioning set.\n        '
        return DataStream(self._j_data_stream.forward())

    @overload
    def broadcast(self) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        pass

    @overload
    def broadcast(self, broadcast_state_descriptor: MapStateDescriptor, *other_broadcast_state_descriptors: MapStateDescriptor) -> 'BroadcastStream':
        if False:
            i = 10
            return i + 15
        pass

    def broadcast(self, broadcast_state_descriptor: Optional[MapStateDescriptor]=None, *other_broadcast_state_descriptors: MapStateDescriptor) -> Union['DataStream', 'BroadcastStream']:
        if False:
            while True:
                i = 10
        '\n        Sets the partitioning of the DataStream so that the output elements are broadcasted to every\n        parallel instance of the next operation.\n\n        If :class:`~state.MapStateDescriptor` s are passed in, it returns a\n        :class:`BroadcastStream` with :class:`~state.BroadcastState` s implicitly created as the\n        descriptors specified.\n\n        Example:\n        ::\n\n            >>> map_state_desc1 = MapStateDescriptor("state1", Types.INT(), Types.INT())\n            >>> map_state_desc2 = MapStateDescriptor("state2", Types.INT(), Types.STRING())\n            >>> broadcast_stream = ds1.broadcast(map_state_desc1, map_state_desc2)\n            >>> broadcast_connected_stream = ds2.connect(broadcast_stream)\n\n        :param broadcast_state_descriptor: the first MapStateDescriptor describing BroadcastState.\n        :param other_broadcast_state_descriptors: the rest of MapStateDescriptors describing\n            BroadcastStates, if any.\n        :return: The DataStream with broadcast partitioning set or a BroadcastStream which can be\n            used in :meth:`connect` to create a BroadcastConnectedStream for further processing of\n            the elements.\n\n        .. versionchanged:: 1.16.0\n           Support return BroadcastStream\n        '
        if broadcast_state_descriptor is not None:
            args = [broadcast_state_descriptor]
            args.extend(other_broadcast_state_descriptors)
            for arg in args:
                if not isinstance(arg, MapStateDescriptor):
                    raise TypeError('broadcast_state_descriptor must be MapStateDescriptor')
            broadcast_state_descriptors = [arg for arg in args]
            return BroadcastStream(cast(DataStream, self.broadcast()), broadcast_state_descriptors)
        return DataStream(self._j_data_stream.broadcast())

    def process(self, func: ProcessFunction, output_type: TypeInformation=None) -> 'DataStream':
        if False:
            return 10
        '\n        Applies the given ProcessFunction on the input stream, thereby creating a transformed output\n        stream.\n\n        The function will be called for every element in the input streams and can produce zero or\n        more output elements.\n\n        :param func: The ProcessFunction that is called for each element in the stream.\n        :param output_type: TypeInformation for the result type of the function.\n        :return: The transformed DataStream.\n        '
        from pyflink.fn_execution import flink_fn_execution_pb2
        (j_python_data_stream_function_operator, j_output_type_info) = _get_one_input_stream_operator(self, func, flink_fn_execution_pb2.UserDefinedDataStreamFunction.PROCESS, output_type)
        return DataStream(self._j_data_stream.transform('PROCESS', j_output_type_info, j_python_data_stream_function_operator))

    def assign_timestamps_and_watermarks(self, watermark_strategy: WatermarkStrategy) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Assigns timestamps to the elements in the data stream and generates watermarks to signal\n        event time progress. The given {@link WatermarkStrategy} is used to create a\n        TimestampAssigner and WatermarkGenerator.\n\n        :param watermark_strategy: The strategy to generate watermarks based on event timestamps.\n        :return: The stream after the transformation, with assigned timestamps and watermarks.\n        '
        if watermark_strategy._timestamp_assigner is not None:

            class TimestampAssignerProcessFunctionAdapter(ProcessFunction):

                def __init__(self, timestamp_assigner: TimestampAssigner):
                    if False:
                        i = 10
                        return i + 15
                    self._extract_timestamp_func = timestamp_assigner.extract_timestamp

                def process_element(self, value, ctx: 'ProcessFunction.Context'):
                    if False:
                        while True:
                            i = 10
                    yield (value, self._extract_timestamp_func(value, ctx.timestamp()))
            timestamped_data_stream = self.process(TimestampAssignerProcessFunctionAdapter(watermark_strategy._timestamp_assigner), Types.TUPLE([self.get_type(), Types.LONG()]))
            timestamped_data_stream.name('Extract-Timestamp')
            gateway = get_gateway()
            JCustomTimestampAssigner = gateway.jvm.org.apache.flink.streaming.api.functions.python.eventtime.CustomTimestampAssigner
            j_watermarked_data_stream = timestamped_data_stream._j_data_stream.assignTimestampsAndWatermarks(watermark_strategy._j_watermark_strategy.withTimestampAssigner(JCustomTimestampAssigner()))
            JRemoveTimestampMapFunction = gateway.jvm.org.apache.flink.streaming.api.functions.python.eventtime.RemoveTimestampMapFunction
            result = DataStream(j_watermarked_data_stream.map(JRemoveTimestampMapFunction(), self._j_data_stream.getType()))
            result.name('Remove-Timestamp')
            return result
        else:
            return DataStream(self._j_data_stream.assignTimestampsAndWatermarks(watermark_strategy._j_watermark_strategy))

    def partition_custom(self, partitioner: Union[Callable, Partitioner], key_selector: Union[Callable, KeySelector]) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        '\n        Partitions a DataStream on the key returned by the selector, using a custom partitioner.\n        This method takes the key selector to get the key to partition on, and a partitioner that\n        accepts the key type.\n\n        Note that this method works only on single field keys, i.e. the selector cannot return\n        tuples of fields.\n\n        :param partitioner: The partitioner to assign partitions to keys.\n        :param key_selector: The KeySelector with which the DataStream is partitioned.\n        :return: The partitioned DataStream.\n        '
        if not isinstance(partitioner, Partitioner) and (not callable(partitioner)):
            raise TypeError('Parameter partitioner should be type of Partitioner or a callable function.')
        if not isinstance(key_selector, KeySelector) and (not callable(key_selector)):
            raise TypeError('Parameter key_selector should be type of KeySelector or a callable function.')
        gateway = get_gateway()

        class CustomPartitioner(ProcessFunction):
            """
            A wrapper class for partition_custom map function. It indicates that it is a partition
            custom operation that we need to apply PythonPartitionCustomOperator
            to run the map function.
            """

            def __init__(self, partitioner, key_selector):
                if False:
                    i = 10
                    return i + 15
                if isinstance(partitioner, Partitioner):
                    self._partitioner_open_func = partitioner.open
                    self._partitioner_close_func = partitioner.close
                    self._partition_func = partitioner.partition
                else:
                    self._partitioner_open_func = None
                    self._partitioner_close_func = None
                    self._partition_func = partitioner
                if isinstance(key_selector, KeySelector):
                    self._key_selector_open_func = key_selector.open
                    self._key_selector_close_func = key_selector.close
                    self._get_key_func = key_selector.get_key
                else:
                    self._key_selector_open_func = None
                    self._key_selector_close_func = None
                    self._get_key_func = key_selector

            def open(self, runtime_context: RuntimeContext):
                if False:
                    print('Hello World!')
                if self._partitioner_open_func:
                    self._partitioner_open_func(runtime_context)
                if self._key_selector_open_func:
                    self._key_selector_open_func(runtime_context)
                self.num_partitions = int(runtime_context.get_job_parameter('NUM_PARTITIONS', '-1'))
                if self.num_partitions <= 0:
                    raise ValueError('The partition number should be a positive value, got %s' % self.num_partitions)

            def close(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self._partitioner_close_func:
                    self._partitioner_close_func()
                if self._key_selector_close_func:
                    self._key_selector_close_func()

            def process_element(self, value, ctx: 'ProcessFunction.Context'):
                if False:
                    return 10
                partition = self._partition_func(self._get_key_func(value), self.num_partitions)
                yield Row(partition, value)
        original_type_info = self.get_type()
        stream_with_partition_info = self.process(CustomPartitioner(partitioner, key_selector), output_type=Types.ROW([Types.INT(), original_type_info]))
        stream_with_partition_info.name(gateway.jvm.org.apache.flink.python.util.PythonConfigUtil.STREAM_PARTITION_CUSTOM_MAP_OPERATOR_NAME)
        JPartitionCustomKeySelector = gateway.jvm.PartitionCustomKeySelector
        JIdParitioner = gateway.jvm.org.apache.flink.api.java.functions.IdPartitioner
        partitioned_stream_with_partition_info = DataStream(stream_with_partition_info._j_data_stream.partitionCustom(JIdParitioner(), JPartitionCustomKeySelector()))
        partitioned_stream = partitioned_stream_with_partition_info.map(lambda x: x[1], original_type_info)
        partitioned_stream.name(gateway.jvm.org.apache.flink.python.util.PythonConfigUtil.KEYED_STREAM_VALUE_OPERATOR_NAME)
        return DataStream(partitioned_stream._j_data_stream)

    def add_sink(self, sink_func: SinkFunction) -> 'DataStreamSink':
        if False:
            i = 10
            return i + 15
        '\n        Adds the given sink to this DataStream. Only streams with sinks added will be executed once\n        the StreamExecutionEnvironment.execute() method is called.\n\n        :param sink_func: The SinkFunction object.\n        :return: The closed DataStream.\n        '
        return DataStreamSink(self._j_data_stream.addSink(sink_func.get_java_function()))

    def sink_to(self, sink: Sink) -> 'DataStreamSink':
        if False:
            while True:
                i = 10
        '\n        Adds the given sink to this DataStream. Only streams with sinks added will be\n        executed once the\n        :func:`~pyflink.datastream.stream_execution_environment.StreamExecutionEnvironment.execute`\n        method is called.\n\n        :param sink: The user defined sink.\n        :return: The closed DataStream.\n        '
        ds = self
        from pyflink.datastream.connectors.base import SupportsPreprocessing
        if isinstance(sink, SupportsPreprocessing) and sink.get_transformer() is not None:
            ds = sink.get_transformer().apply(self)
        return DataStreamSink(ds._j_data_stream.sinkTo(sink.get_java_function()))

    def execute_and_collect(self, job_execution_name: str=None, limit: int=None) -> Union['CloseableIterator', list]:
        if False:
            return 10
        "\n        Triggers the distributed execution of the streaming dataflow and returns an iterator over\n        the elements of the given DataStream.\n\n        The DataStream application is executed in the regular distributed manner on the target\n        environment, and the events from the stream are polled back to this application process and\n        thread through Flink's REST API.\n\n        The returned iterator must be closed to free all cluster resources.\n\n        :param job_execution_name: The name of the job execution.\n        :param limit: The limit for the collected elements.\n        "
        JPythonConfigUtil = get_gateway().jvm.org.apache.flink.python.util.PythonConfigUtil
        JPythonConfigUtil.configPythonOperator(self._j_data_stream.getExecutionEnvironment())
        self._apply_chaining_optimization()
        if job_execution_name is None and limit is None:
            return CloseableIterator(self._j_data_stream.executeAndCollect(), self.get_type())
        elif job_execution_name is not None and limit is None:
            return CloseableIterator(self._j_data_stream.executeAndCollect(job_execution_name), self.get_type())
        if job_execution_name is None and limit is not None:
            return list(map(lambda data: convert_to_python_obj(data, self.get_type()), self._j_data_stream.executeAndCollect(limit)))
        else:
            return list(map(lambda data: convert_to_python_obj(data, self.get_type()), self._j_data_stream.executeAndCollect(job_execution_name, limit)))

    def print(self, sink_identifier: str=None) -> 'DataStreamSink':
        if False:
            return 10
        '\n        Writes a DataStream to the standard output stream (stdout).\n        For each element of the DataStream the object string is written.\n\n        NOTE: This will print to stdout on the machine where the code is executed, i.e. the Flink\n        worker, and is not fault tolerant.\n\n        :param sink_identifier: The string to prefix the output with.\n        :return: The closed DataStream.\n        '
        if sink_identifier is not None:
            j_data_stream_sink = self._align_output_type()._j_data_stream.print(sink_identifier)
        else:
            j_data_stream_sink = self._align_output_type()._j_data_stream.print()
        return DataStreamSink(j_data_stream_sink)

    def get_side_output(self, output_tag: OutputTag) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Gets the :class:`DataStream` that contains the elements that are emitted from an operation\n        into the side output with the given :class:`OutputTag`.\n\n        :param output_tag: output tag for the side stream\n        :return: The DataStream with specified output tag\n\n        .. versionadded:: 1.16.0\n        '
        ds = DataStream(self._j_data_stream.getSideOutput(output_tag.get_java_output_tag()))
        return ds.map(lambda i: i, output_type=output_tag.type_info)

    def cache(self) -> 'CachedDataStream':
        if False:
            print('Hello World!')
        '\n        Cache the intermediate result of the transformation. Only support bounded streams and\n        currently only block mode is supported. The cache is generated lazily at the first time the\n        intermediate result is computed. The cache will be clear when the StreamExecutionEnvironment\n        close.\n\n        :return: The cached DataStream that can use in later job to reuse the cached intermediate\n                 result.\n\n        .. versionadded:: 1.16.0\n        '
        return CachedDataStream(self._j_data_stream.cache())

    def _apply_chaining_optimization(self):
        if False:
            i = 10
            return i + 15
        '\n        Chain the Python operators if possible.\n        '
        gateway = get_gateway()
        JPythonOperatorChainingOptimizer = gateway.jvm.org.apache.flink.python.chain.PythonOperatorChainingOptimizer
        j_transformation = JPythonOperatorChainingOptimizer.apply(self._j_data_stream.getExecutionEnvironment(), self._j_data_stream.getTransformation())
        self._j_data_stream = gateway.jvm.org.apache.flink.streaming.api.datastream.DataStream(self._j_data_stream.getExecutionEnvironment(), j_transformation)

    def _align_output_type(self) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Transform the pickled python object into String if the output type is PickledByteArrayInfo.\n        '
        from py4j.java_gateway import get_java_class
        gateway = get_gateway()
        ExternalTypeInfo_CLASS = get_java_class(gateway.jvm.org.apache.flink.table.runtime.typeutils.ExternalTypeInfo)
        RowTypeInfo_CLASS = get_java_class(gateway.jvm.org.apache.flink.api.java.typeutils.RowTypeInfo)
        output_type_info_class = self._j_data_stream.getTransformation().getOutputType().getClass()
        if output_type_info_class.isAssignableFrom(Types.PICKLED_BYTE_ARRAY().get_java_type_info().getClass()):

            def python_obj_to_str_map_func(value):
                if False:
                    print('Hello World!')
                if not isinstance(value, (str, bytes)):
                    value = str(value)
                return value
            transformed_data_stream = DataStream(self.map(python_obj_to_str_map_func, output_type=Types.STRING())._j_data_stream)
            return transformed_data_stream
        elif output_type_info_class.isAssignableFrom(ExternalTypeInfo_CLASS) or output_type_info_class.isAssignableFrom(RowTypeInfo_CLASS):

            def python_obj_to_str_map_func(value):
                if False:
                    i = 10
                    return i + 15
                assert isinstance(value, Row)
                return '{}[{}]'.format(value.get_row_kind(), ','.join([str(item) for item in value._values]))
            transformed_data_stream = DataStream(self.map(python_obj_to_str_map_func, output_type=Types.STRING())._j_data_stream)
            return transformed_data_stream
        else:
            return self

class DataStreamSink(object):
    """
    A Stream Sink. This is used for emitting elements from a streaming topology.
    """

    def __init__(self, j_data_stream_sink):
        if False:
            for i in range(10):
                print('nop')
        '\n        The constructor of DataStreamSink.\n\n        :param j_data_stream_sink: A DataStreamSink java object.\n        '
        self._j_data_stream_sink = j_data_stream_sink

    def name(self, name: str) -> 'DataStreamSink':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the name of this sink. THis name is used by the visualization and logging during\n        runtime.\n\n        :param name: The name of this sink.\n        :return: The named sink.\n        '
        self._j_data_stream_sink.name(name)
        return self

    def uid(self, uid: str) -> 'DataStreamSink':
        if False:
            while True:
                i = 10
        '\n        Sets an ID for this operator. The specified ID is used to assign the same operator ID across\n        job submissions (for example when starting a job from a savepoint).\n\n        Important: this ID needs to be unique per transformation and job. Otherwise, job submission\n        will fail.\n\n        :param uid: The unique user-specified ID of this transformation.\n        :return: The operator with the specified ID.\n        '
        self._j_data_stream_sink.uid(uid)
        return self

    def set_uid_hash(self, uid_hash: str) -> 'DataStreamSink':
        if False:
            print('Hello World!')
        '\n        Sets an user provided hash for this operator. This will be used AS IS the create the\n        JobVertexID. The user provided hash is an alternative to the generated hashed, that is\n        considered when identifying an operator through the default hash mechanics fails (e.g.\n        because of changes between Flink versions).\n\n        Important: this should be used as a workaround or for trouble shooting. The provided hash\n        needs to be unique per transformation and job. Otherwise, job submission will fail.\n        Furthermore, you cannot assign user-specified hash to intermediate nodes in an operator\n        chain and trying so will let your job fail.\n\n        A use case for this is in migration between Flink versions or changing the jobs in a way\n        that changes the automatically generated hashes. In this case, providing the previous hashes\n        directly through this method (e.g. obtained from old logs) can help to reestablish a lost\n        mapping from states to their target operator.\n\n        :param uid_hash: The user provided hash for this operator. This will become the jobVertexID,\n                         which is shown in the logs and web ui.\n        :return: The operator with the user provided hash.\n        '
        self._j_data_stream_sink.setUidHash(uid_hash)
        return self

    def set_parallelism(self, parallelism: int) -> 'DataStreamSink':
        if False:
            print('Hello World!')
        '\n        Sets the parallelism for this operator.\n\n        :param parallelism: THe parallelism for this operator.\n        :return: The operator with set parallelism.\n        '
        self._j_data_stream_sink.setParallelism(parallelism)
        return self

    def set_description(self, description: str) -> 'DataStreamSink':
        if False:
            return 10
        '\n        Sets the description for this sink.\n\n        Description is used in json plan and web ui, but not in logging and metrics where only\n        name is available. Description is expected to provide detailed information about the sink,\n        while name is expected to be more simple, providing summary information only, so that we can\n        have more user-friendly logging messages and metric tags without losing useful messages for\n        debugging.\n\n        :param description: The description for this sink.\n        :return: The sink with new description.\n\n        .. versionadded:: 1.15.0\n        '
        self._j_data_stream_sink.setDescription(description)
        return self

    def disable_chaining(self) -> 'DataStreamSink':
        if False:
            print('Hello World!')
        '\n        Turns off chaining for this operator so thread co-location will not be used as an\n        optimization.\n        Chaining can be turned off for the whole job by\n        StreamExecutionEnvironment.disableOperatorChaining() however it is not advised for\n        performance consideration.\n\n        :return: The operator with chaining disabled.\n        '
        self._j_data_stream_sink.disableChaining()
        return self

    def slot_sharing_group(self, slot_sharing_group: Union[str, SlotSharingGroup]) -> 'DataStreamSink':
        if False:
            return 10
        "\n        Sets the slot sharing group of this operation. Parallel instances of operations that are in\n        the same slot sharing group will be co-located in the same TaskManager slot, if possible.\n\n        Operations inherit the slot sharing group of input operations if all input operations are in\n        the same slot sharing group and no slot sharing group was explicitly specified.\n\n        Initially an operation is in the default slot sharing group. An operation can be put into\n        the default group explicitly by setting the slot sharing group to 'default'.\n\n        :param slot_sharing_group: The slot sharing group name or which contains name and its\n                        resource spec.\n        :return: This operator.\n        "
        if isinstance(slot_sharing_group, SlotSharingGroup):
            self._j_data_stream_sink.slotSharingGroup(slot_sharing_group.get_java_slot_sharing_group())
        else:
            self._j_data_stream_sink.slotSharingGroup(slot_sharing_group)
        return self

class KeyedStream(DataStream):
    """
    A KeyedStream represents a DataStream on which operator state is partitioned by key using a
    provided KeySelector. Typical operations supported by a DataStream are also possible on a
    KeyedStream, with the exception of partitioning methods such as shuffle, forward and keyBy.

    Reduce-style operations, such as reduce and sum work on elements that have the same key.
    """

    def __init__(self, j_keyed_stream, original_data_type_info, origin_stream: DataStream):
        if False:
            while True:
                i = 10
        '\n        Constructor of KeyedStream.\n\n        :param j_keyed_stream: A java KeyedStream object.\n        :param original_data_type_info: Original data typeinfo.\n        :param origin_stream: The DataStream before key by.\n        '
        super(KeyedStream, self).__init__(j_data_stream=j_keyed_stream)
        self._original_data_type_info = original_data_type_info
        self._origin_stream = origin_stream

    def map(self, func: Union[Callable, MapFunction], output_type: TypeInformation=None) -> 'DataStream':
        if False:
            while True:
                i = 10
        '\n        Applies a Map transformation on a KeyedStream. The transformation calls a MapFunction for\n        each element of the DataStream. Each MapFunction call returns exactly one element.\n\n        Note that If user does not specify the output data type, the output data will be serialized\n        as pickle primitive byte array.\n\n        :param func: The MapFunction that is called for each element of the DataStream.\n        :param output_type: The type information of the MapFunction output data.\n        :return: The transformed DataStream.\n        '
        if not isinstance(func, MapFunction) and (not callable(func)):
            raise TypeError('The input must be a MapFunction or a callable function')

        class MapKeyedProcessFunctionAdapter(KeyedProcessFunction):

            def __init__(self, map_func):
                if False:
                    print('Hello World!')
                if isinstance(map_func, MapFunction):
                    self._open_func = map_func.open
                    self._close_func = map_func.close
                    self._map_func = map_func.map
                else:
                    self._open_func = None
                    self._close_func = None
                    self._map_func = map_func

            def open(self, runtime_context: RuntimeContext):
                if False:
                    for i in range(10):
                        print('nop')
                if self._open_func:
                    self._open_func(runtime_context)

            def close(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self._close_func:
                    self._close_func()

            def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
                if False:
                    i = 10
                    return i + 15
                yield self._map_func(value)
        return self.process(MapKeyedProcessFunctionAdapter(func), output_type).name('Map')

    def flat_map(self, func: Union[Callable, FlatMapFunction], output_type: TypeInformation=None) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Applies a FlatMap transformation on a KeyedStream. The transformation calls a\n        FlatMapFunction for each element of the DataStream. Each FlatMapFunction call can return\n        any number of elements including none.\n\n        :param func: The FlatMapFunction that is called for each element of the DataStream.\n        :param output_type: The type information of output data.\n        :return: The transformed DataStream.\n        '
        if not isinstance(func, FlatMapFunction) and (not callable(func)):
            raise TypeError('The input must be a FlatMapFunction or a callable function')

        class FlatMapKeyedProcessFunctionAdapter(KeyedProcessFunction):

            def __init__(self, flat_map_func):
                if False:
                    while True:
                        i = 10
                if isinstance(flat_map_func, FlatMapFunction):
                    self._open_func = flat_map_func.open
                    self._close_func = flat_map_func.close
                    self._flat_map_func = flat_map_func.flat_map
                else:
                    self._open_func = None
                    self._close_func = None
                    self._flat_map_func = flat_map_func

            def open(self, runtime_context: RuntimeContext):
                if False:
                    for i in range(10):
                        print('nop')
                if self._open_func:
                    self._open_func(runtime_context)

            def close(self):
                if False:
                    while True:
                        i = 10
                if self._close_func:
                    self._close_func()

            def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
                if False:
                    for i in range(10):
                        print('nop')
                yield from self._flat_map_func(value)
        return self.process(FlatMapKeyedProcessFunctionAdapter(func), output_type).name('FlatMap')

    def reduce(self, func: Union[Callable, ReduceFunction]) -> 'DataStream':
        if False:
            while True:
                i = 10
        "\n        Applies a reduce transformation on the grouped data stream grouped on by the given\n        key position. The `ReduceFunction` will receive input values based on the key value.\n        Only input values with the same key will go to the same reducer.\n\n        Example:\n        ::\n\n            >>> ds = env.from_collection([(1, 'a'), (2, 'a'), (3, 'a'), (4, 'b'])\n            >>> ds.key_by(lambda x: x[1]).reduce(lambda a, b: a[0] + b[0], b[1])\n\n        :param func: The ReduceFunction that is called for each element of the DataStream.\n        :return: The transformed DataStream.\n        "
        if not isinstance(func, ReduceFunction) and (not callable(func)):
            raise TypeError('The input must be a ReduceFunction or a callable function')
        output_type = _from_java_type(self._original_data_type_info.get_java_type_info())
        gateway = get_gateway()
        j_conf = get_j_env_configuration(self._j_data_stream.getExecutionEnvironment())
        python_execution_mode = j_conf.getString(gateway.jvm.org.apache.flink.python.PythonOptions.PYTHON_EXECUTION_MODE)

        class ReduceProcessKeyedProcessFunctionAdapter(KeyedProcessFunction):

            def __init__(self, reduce_function):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(reduce_function, ReduceFunction):
                    self._open_func = reduce_function.open
                    self._close_func = reduce_function.close
                    self._reduce_function = reduce_function.reduce
                else:
                    self._open_func = None
                    self._close_func = None
                    self._reduce_function = reduce_function
                self._reduce_state = None
                self._in_batch_execution_mode = True

            def open(self, runtime_context: RuntimeContext):
                if False:
                    i = 10
                    return i + 15
                if self._open_func:
                    self._open_func(runtime_context)
                self._reduce_state = runtime_context.get_reducing_state(ReducingStateDescriptor('_reduce_state' + str(uuid.uuid4()), self._reduce_function, output_type))
                if python_execution_mode == 'process':
                    from pyflink.fn_execution.datastream.process.runtime_context import StreamingRuntimeContext
                    self._in_batch_execution_mode = cast(StreamingRuntimeContext, runtime_context)._in_batch_execution_mode
                else:
                    self._in_batch_execution_mode = runtime_context.get_job_parameter('inBatchExecutionMode', 'false') == 'true'

            def close(self):
                if False:
                    i = 10
                    return i + 15
                if self._close_func:
                    self._close_func()

            def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
                if False:
                    while True:
                        i = 10
                if self._in_batch_execution_mode:
                    reduce_value = self._reduce_state.get()
                    if reduce_value is None:
                        ctx.timer_service().register_event_time_timer(9223372036854775807)
                    self._reduce_state.add(value)
                else:
                    self._reduce_state.add(value)
                    yield self._reduce_state.get()

            def on_timer(self, timestamp: int, ctx: 'KeyedProcessFunction.OnTimerContext'):
                if False:
                    while True:
                        i = 10
                current_value = self._reduce_state.get()
                if current_value is not None:
                    yield current_value
        return self.process(ReduceProcessKeyedProcessFunctionAdapter(func), output_type).name('Reduce')

    def filter(self, func: Union[Callable, FilterFunction]) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        if not isinstance(func, FilterFunction) and (not callable(func)):
            raise TypeError('The input must be a FilterFunction or a callable function')

        class FilterKeyedProcessFunctionAdapter(KeyedProcessFunction):

            def __init__(self, filter_func):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(filter_func, FilterFunction):
                    self._open_func = filter_func.open
                    self._close_func = filter_func.close
                    self._filter_func = filter_func.filter
                else:
                    self._open_func = None
                    self._close_func = None
                    self._filter_func = filter_func

            def open(self, runtime_context: RuntimeContext):
                if False:
                    for i in range(10):
                        print('nop')
                if self._open_func:
                    self._open_func(runtime_context)

            def close(self):
                if False:
                    print('Hello World!')
                if self._close_func:
                    self._close_func()

            def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
                if False:
                    i = 10
                    return i + 15
                if self._filter_func(value):
                    yield value
        return self.process(FilterKeyedProcessFunctionAdapter(func), self._original_data_type_info).name('Filter')

    class AccumulateType(Enum):
        MIN = 1
        MAX = 2
        MIN_BY = 3
        MAX_BY = 4
        SUM = 5

    def _accumulate(self, position: Union[int, str], acc_type: AccumulateType):
        if False:
            while True:
                i = 10
        '\n        The base method is used for operators such as min, max, min_by, max_by, sum.\n        '
        if not isinstance(position, int) and (not isinstance(position, str)):
            raise TypeError('The field position must be of int or str type to locate the value to calculate for min, max, min_by, max_by and sum.The given type is: %s' % type(position))

        class AccumulateReduceFunction(ReduceFunction):

            def __init__(self, position, agg_type):
                if False:
                    for i in range(10):
                        print('nop')
                self._pos = position
                self._agg_type = agg_type
                self._reduce_func = None

            def reduce(self, value1, value2):
                if False:
                    for i in range(10):
                        print('nop')

                def init_reduce_func(value_to_check):
                    if False:
                        while True:
                            i = 10
                    if acc_type == KeyedStream.AccumulateType.MIN_BY:

                        def reduce_func(v1, v2):
                            if False:
                                i = 10
                                return i + 15
                            if isinstance(value_to_check, (tuple, list, Row)):
                                return v2 if v2[self._pos] < v1[self._pos] else v1
                            else:
                                return v2 if v2 < v1 else v1
                        self._reduce_func = reduce_func
                    elif acc_type == KeyedStream.AccumulateType.MAX_BY:

                        def reduce_func(v1, v2):
                            if False:
                                i = 10
                                return i + 15
                            if isinstance(value_to_check, (tuple, list, Row)):
                                return v2 if v2[self._pos] > v1[self._pos] else v1
                            else:
                                return v2 if v2 > v1 else v1
                        self._reduce_func = reduce_func
                    elif isinstance(value_to_check, tuple):

                        def reduce_func(v1, v2):
                            if False:
                                return 10
                            v1_list = list(v1)
                            if acc_type == KeyedStream.AccumulateType.MIN:
                                v1_list[self._pos] = v2[self._pos] if v2[self._pos] < v1[self._pos] else v1[self._pos]
                            elif acc_type == KeyedStream.AccumulateType.MAX:
                                v1_list[self._pos] = v2[self._pos] if v2[self._pos] > v1[self._pos] else v1[self._pos]
                            else:
                                v1_list[self._pos] = v1[self._pos] + v2[self._pos]
                                return tuple(v1_list)
                            return tuple(v1_list)
                        self._reduce_func = reduce_func
                    elif isinstance(value_to_check, (list, Row)):

                        def reduce_func(v1, v2):
                            if False:
                                return 10
                            if acc_type == KeyedStream.AccumulateType.MIN:
                                v1[self._pos] = v2[self._pos] if v2[self._pos] < v1[self._pos] else v1[self._pos]
                            elif acc_type == KeyedStream.AccumulateType.MAX:
                                v1[self._pos] = v2[self._pos] if v2[self._pos] > v1[self._pos] else v1[self._pos]
                            else:
                                v1[self._pos] = v1[self._pos] + v2[self._pos]
                            return v1
                        self._reduce_func = reduce_func
                    else:
                        if self._pos != 0:
                            raise TypeError('The %s field selected on a basic type. A field expression on a basic type can only select the 0th field (which means selecting the entire basic type).' % self._pos)

                        def reduce_func(v1, v2):
                            if False:
                                i = 10
                                return i + 15
                            if acc_type == KeyedStream.AccumulateType.MIN:
                                return v2 if v2 < v1 else v1
                            elif acc_type == KeyedStream.AccumulateType.MAX:
                                return v2 if v2 > v1 else v1
                            else:
                                return v1 + v2
                        self._reduce_func = reduce_func
                if not self._reduce_func:
                    init_reduce_func(value2)
                return self._reduce_func(value1, value2)
        return self.reduce(AccumulateReduceFunction(position, acc_type))

    def sum(self, position_to_sum: Union[int, str]=0) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        '\n        Applies an aggregation that gives a rolling sum of the data stream at the given position\n        grouped by the given key. An independent aggregate is kept per key.\n\n        Example(Tuple data to sum):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'b\', 1), (\'b\', 5)])\n            >>> ds.key_by(lambda x: x[0]).sum(1)\n\n        Example(Row data to sum):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...                          type_info=Types.ROW([Types.STRING(), Types.INT()]))\n            >>> ds.key_by(lambda x: x[0]).sum(1)\n\n        Example(Row data with fields name to sum):\n        ::\n\n            >>> ds = env.from_collection(\n            ...     [(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...     type_info=Types.ROW_NAMED(["key", "value"], [Types.STRING(), Types.INT()])\n            ... )\n            >>> ds.key_by(lambda x: x[0]).sum("value")\n\n        :param position_to_sum: The field position in the data points to sum, type can be int which\n                                indicates the index of the column to operate on or str which\n                                indicates the name of the column to operate on.\n        :return: The transformed DataStream.\n\n        .. versionadded:: 1.16.0\n        '
        return self._accumulate(position_to_sum, KeyedStream.AccumulateType.SUM)

    def min(self, position_to_min: Union[int, str]=0) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Applies an aggregation that gives the current minimum of the data stream at the given\n        position by the given key. An independent aggregate is kept per key.\n\n        Example(Tuple data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'b\', 1), (\'b\', 5)])\n            >>> ds.key_by(lambda x: x[0]).min(1)\n\n        Example(Row data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...                          type_info=Types.ROW([Types.STRING(), Types.INT()]))\n            >>> ds.key_by(lambda x: x[0]).min(1)\n\n        Example(Row data with fields name):\n        ::\n\n            >>> ds = env.from_collection(\n            ...     [(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...     type_info=Types.ROW_NAMED(["key", "value"], [Types.STRING(), Types.INT()])\n            ... )\n            >>> ds.key_by(lambda x: x[0]).min("value")\n\n        :param position_to_min: The field position in the data points to minimize. The type can be\n                                int (field position) or str (field name). This is applicable to\n                                Tuple types, List types, Row types, and basic types (which is\n                                considered as having one field).\n        :return: The transformed DataStream.\n\n        .. versionadded:: 1.16.0\n        '
        return self._accumulate(position_to_min, KeyedStream.AccumulateType.MIN)

    def max(self, position_to_max: Union[int, str]=0) -> 'DataStream':
        if False:
            return 10
        '\n        Applies an aggregation that gives the current maximize of the data stream at the given\n        position by the given key. An independent aggregate is kept per key.\n\n        Example(Tuple data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'b\', 1), (\'b\', 5)])\n            >>> ds.key_by(lambda x: x[0]).max(1)\n\n        Example(Row data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...                          type_info=Types.ROW([Types.STRING(), Types.INT()]))\n            >>> ds.key_by(lambda x: x[0]).max(1)\n\n        Example(Row data with fields name):\n        ::\n\n            >>> ds = env.from_collection(\n            ...     [(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...     type_info=Types.ROW_NAMED(["key", "value"], [Types.STRING(), Types.INT()])\n            ... )\n            >>> ds.key_by(lambda x: x[0]).max("value")\n\n        :param position_to_max: The field position in the data points to maximize. The type can be\n                                int (field position) or str (field name). This is applicable to\n                                Tuple types, List types, Row types, and basic types (which is\n                                considered as having one field).\n        :return: The transformed DataStream.\n\n        .. versionadded:: 1.16.0\n        '
        return self._accumulate(position_to_max, KeyedStream.AccumulateType.MAX)

    def min_by(self, position_to_min_by: Union[int, str]=0) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Applies an aggregation that gives the current element with the minimum value at the\n        given position by the given key. An independent aggregate is kept per key.\n        If more elements have the minimum value at the given position,\n        the operator returns the first one by default.\n\n        Example(Tuple data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'b\', 1), (\'b\', 5)])\n            >>> ds.key_by(lambda x: x[0]).min_by(1)\n\n        Example(Row data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...                          type_info=Types.ROW([Types.STRING(), Types.INT()]))\n            >>> ds.key_by(lambda x: x[0]).min_by(1)\n\n        Example(Row data with fields name):\n        ::\n\n            >>> ds = env.from_collection(\n            ...     [(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...     type_info=Types.ROW_NAMED(["key", "value"], [Types.STRING(), Types.INT()])\n            ... )\n            >>> ds.key_by(lambda x: x[0]).min_by("value")\n\n        :param position_to_min_by: The field position in the data points to minimize. The type can\n                                   be int (field position) or str (field name). This is applicable\n                                   to Tuple types, List types, Row types, and basic types (which is\n                                   considered as having one field).\n        :return: The transformed DataStream.\n\n        .. versionadded:: 1.16.0\n        '
        return self._accumulate(position_to_min_by, KeyedStream.AccumulateType.MIN_BY)

    def max_by(self, position_to_max_by: Union[int, str]=0) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        '\n        Applies an aggregation that gives the current element with the maximize value at the\n        given position by the given key. An independent aggregate is kept per key.\n        If more elements have the maximize value at the given position,\n        the operator returns the first one by default.\n\n\n        Example(Tuple data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'b\', 1), (\'b\', 5)])\n            >>> ds.key_by(lambda x: x[0]).max_by(1)\n\n        Example(Row data):\n        ::\n\n            >>> ds = env.from_collection([(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...                          type_info=Types.ROW([Types.STRING(), Types.INT()]))\n            >>> ds.key_by(lambda x: x[0]).max_by(1)\n\n        Example(Row data with fields name):\n        ::\n\n            >>> ds = env.from_collection(\n            ...     [(\'a\', 1), (\'a\', 2), (\'a\', 3), (\'b\', 1), (\'b\', 2)],\n            ...     type_info=Types.ROW_NAMED(["key", "value"], [Types.STRING(), Types.INT()])\n            ... )\n            >>> ds.key_by(lambda x: x[0]).max_by("value")\n\n        :param position_to_max_by: The field position in the data points to maximize. The type can\n                                   be int (field position) or str (field name). This is applicable\n                                   to Tuple types, List types, Row types, and basic types (which is\n                                   considered as having one field).\n        :return: The transformed DataStream.\n\n        .. versionadded:: 1.16.0\n        '
        return self._accumulate(position_to_max_by, KeyedStream.AccumulateType.MAX_BY)

    def add_sink(self, sink_func: SinkFunction) -> 'DataStreamSink':
        if False:
            i = 10
            return i + 15
        return self._values().add_sink(sink_func)

    def key_by(self, key_selector: Union[Callable, KeySelector], key_type: TypeInformation=None) -> 'KeyedStream':
        if False:
            return 10
        return self._origin_stream.key_by(key_selector, key_type)

    def process(self, func: KeyedProcessFunction, output_type: TypeInformation=None) -> 'DataStream':
        if False:
            return 10
        '\n        Applies the given ProcessFunction on the input stream, thereby creating a transformed output\n        stream.\n\n        The function will be called for every element in the input streams and can produce zero or\n        more output elements.\n\n        :param func: The KeyedProcessFunction that is called for each element in the stream.\n        :param output_type: TypeInformation for the result type of the function.\n        :return: The transformed DataStream.\n        '
        if not isinstance(func, KeyedProcessFunction):
            raise TypeError('KeyedProcessFunction is required for KeyedStream.')
        from pyflink.fn_execution import flink_fn_execution_pb2
        (j_python_data_stream_function_operator, j_output_type_info) = _get_one_input_stream_operator(self, func, flink_fn_execution_pb2.UserDefinedDataStreamFunction.KEYED_PROCESS, output_type)
        return DataStream(self._j_data_stream.transform('KEYED PROCESS', j_output_type_info, j_python_data_stream_function_operator))

    def window(self, window_assigner: WindowAssigner) -> 'WindowedStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Windows this data stream to a WindowedStream, which evaluates windows over a key\n        grouped stream. Elements are put into windows by a WindowAssigner. The grouping of\n        elements is done both by key and by window.\n\n        A Trigger can be defined to specify when windows are evaluated. However, WindowAssigners\n        have a default Trigger that is used if a Trigger is not specified.\n\n        :param window_assigner: The WindowAssigner that assigns elements to windows.\n        :return: The trigger windows data stream.\n        '
        return WindowedStream(self, window_assigner)

    def count_window(self, size: int, slide: int=0):
        if False:
            return 10
        '\n        Windows this KeyedStream into tumbling or sliding count windows.\n\n        :param size: The size of the windows in number of elements.\n        :param slide: The slide interval in number of elements.\n\n        .. versionadded:: 1.16.0\n        '
        if slide == 0:
            return WindowedStream(self, CountTumblingWindowAssigner(size))
        else:
            return WindowedStream(self, CountSlidingWindowAssigner(size, slide))

    def union(self, *streams) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        return self._values().union(*streams)

    @overload
    def connect(self, ds: 'DataStream') -> 'ConnectedStreams':
        if False:
            for i in range(10):
                print('nop')
        pass

    @overload
    def connect(self, ds: 'BroadcastStream') -> 'BroadcastConnectedStream':
        if False:
            for i in range(10):
                print('nop')
        pass

    def connect(self, ds: Union['DataStream', 'BroadcastStream']) -> Union['ConnectedStreams', 'BroadcastConnectedStream']:
        if False:
            return 10
        '\n        If ds is a :class:`DataStream`, creates a new :class:`ConnectedStreams` by connecting\n        DataStream outputs of (possible) different types with each other. The DataStreams connected\n        using this operator can be used with CoFunctions to apply joint transformations.\n\n        If ds is a :class:`BroadcastStream`, creates a new :class:`BroadcastConnectedStream` by\n        connecting the current :class:`DataStream` with a :class:`BroadcastStream`. The latter can\n        be created using the :meth:`broadcast` method. The resulting stream can be further processed\n        using the :meth:`BroadcastConnectedStream.process` method.\n\n        :param ds: The DataStream or BroadcastStream with which this stream will be connected.\n        :return: The ConnectedStreams or BroadcastConnectedStream.\n\n        .. versionchanged:: 1.16.0\n           Support connect BroadcastStream\n        '
        return super().connect(ds)

    def shuffle(self) -> 'DataStream':
        if False:
            print('Hello World!')
        raise Exception('Cannot override partitioning for KeyedStream.')

    def project(self, *field_indexes) -> 'DataStream':
        if False:
            print('Hello World!')
        return self._values().project(*field_indexes)

    def rescale(self) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        raise Exception('Cannot override partitioning for KeyedStream.')

    def rebalance(self) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Cannot override partitioning for KeyedStream.')

    def forward(self) -> 'DataStream':
        if False:
            return 10
        raise Exception('Cannot override partitioning for KeyedStream.')

    def broadcast(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        Not supported, partitioning for KeyedStream cannot be overridden.\n        '
        raise Exception('Cannot override partitioning for KeyedStream.')

    def partition_custom(self, partitioner: Union[Callable, Partitioner], key_selector: Union[Callable, KeySelector]) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Cannot override partitioning for KeyedStream.')

    def print(self, sink_identifier=None):
        if False:
            while True:
                i = 10
        return self._values().print()

    def _values(self) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Since python KeyedStream is in the format of Row(key_value, original_data), it is used for\n        getting the original_data.\n        '
        transformed_stream = self.map(lambda x: x, output_type=self._original_data_type_info)
        transformed_stream.name(get_gateway().jvm.org.apache.flink.python.util.PythonConfigUtil.KEYED_STREAM_VALUE_OPERATOR_NAME)
        return DataStream(transformed_stream._j_data_stream)

    def set_parallelism(self, parallelism: int):
        if False:
            print('Hello World!')
        raise Exception('Set parallelism for KeyedStream is not supported.')

    def name(self, name: str):
        if False:
            i = 10
            return i + 15
        raise Exception('Set name for KeyedStream is not supported.')

    def get_name(self) -> str:
        if False:
            i = 10
            return i + 15
        raise Exception('Get name of KeyedStream is not supported.')

    def uid(self, uid: str):
        if False:
            while True:
                i = 10
        raise Exception('Set uid for KeyedStream is not supported.')

    def set_uid_hash(self, uid_hash: str):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Set uid hash for KeyedStream is not supported.')

    def set_max_parallelism(self, max_parallelism: int):
        if False:
            return 10
        raise Exception('Set max parallelism for KeyedStream is not supported.')

    def force_non_parallel(self):
        if False:
            return 10
        raise Exception('Set force non-parallel for KeyedStream is not supported.')

    def set_buffer_timeout(self, timeout_millis: int):
        if False:
            print('Hello World!')
        raise Exception('Set buffer timeout for KeyedStream is not supported.')

    def start_new_chain(self) -> 'DataStream':
        if False:
            print('Hello World!')
        raise Exception('Start new chain for KeyedStream is not supported.')

    def disable_chaining(self) -> 'DataStream':
        if False:
            return 10
        raise Exception('Disable chaining for KeyedStream is not supported.')

    def slot_sharing_group(self, slot_sharing_group: Union[str, SlotSharingGroup]) -> 'DataStream':
        if False:
            while True:
                i = 10
        raise Exception('Setting slot sharing group for KeyedStream is not supported.')

    def cache(self) -> 'CachedDataStream':
        if False:
            i = 10
            return i + 15
        raise Exception('Cache for KeyedStream is not supported.')

class CachedDataStream(DataStream):
    """
    CachedDataStream represents a DataStream whose intermediate result will be cached at the first
    time when it is computed. And the cached intermediate result can be used in later job that using
    the same CachedDataStream to avoid re-computing the intermediate result.
    """

    def __init__(self, j_data_stream):
        if False:
            i = 10
            return i + 15
        super(CachedDataStream, self).__init__(j_data_stream)

    def invalidate(self):
        if False:
            return 10
        '\n        Invalidate the cache intermediate result of this DataStream to release the physical\n        resources. Users are not required to invoke this method to release physical resources unless\n        they want to. Cache will be recreated if it is used after invalidated.\n\n        .. versionadded:: 1.16.0\n        '
        self._j_data_stream.invalidate()

    def set_parallelism(self, parallelism: int):
        if False:
            while True:
                i = 10
        raise Exception('Set parallelism for CachedDataStream is not supported.')

    def name(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Set name for CachedDataStream is not supported.')

    def get_name(self) -> str:
        if False:
            return 10
        raise Exception('Get name of CachedDataStream is not supported.')

    def uid(self, uid: str):
        if False:
            print('Hello World!')
        raise Exception('Set uid for CachedDataStream is not supported.')

    def set_uid_hash(self, uid_hash: str):
        if False:
            print('Hello World!')
        raise Exception('Set uid hash for CachedDataStream is not supported.')

    def set_max_parallelism(self, max_parallelism: int):
        if False:
            i = 10
            return i + 15
        raise Exception('Set max parallelism for CachedDataStream is not supported.')

    def force_non_parallel(self):
        if False:
            while True:
                i = 10
        raise Exception('Set force non-parallel for CachedDataStream is not supported.')

    def set_buffer_timeout(self, timeout_millis: int):
        if False:
            print('Hello World!')
        raise Exception('Set buffer timeout for CachedDataStream is not supported.')

    def start_new_chain(self) -> 'DataStream':
        if False:
            return 10
        raise Exception('Start new chain for CachedDataStream is not supported.')

    def disable_chaining(self) -> 'DataStream':
        if False:
            return 10
        raise Exception('Disable chaining for CachedDataStream is not supported.')

    def slot_sharing_group(self, slot_sharing_group: Union[str, SlotSharingGroup]) -> 'DataStream':
        if False:
            while True:
                i = 10
        raise Exception('Setting slot sharing group for CachedDataStream is not supported.')

class WindowedStream(object):
    """
    A WindowedStream represents a data stream where elements are grouped by key, and for each
    key, the stream of elements is split into windows based on a WindowAssigner. Window emission
    is triggered based on a Trigger.

    The windows are conceptually evaluated for each key individually, meaning windows can trigger
    at different points for each key.

    Note that the WindowedStream is purely an API construct, during runtime the WindowedStream will
    be collapsed together with the KeyedStream and the operation over the window into one single
    operation.
    """

    def __init__(self, keyed_stream: KeyedStream, window_assigner: WindowAssigner):
        if False:
            while True:
                i = 10
        self._keyed_stream = keyed_stream
        self._window_assigner = window_assigner
        self._allowed_lateness = 0
        self._late_data_output_tag = None
        self._window_trigger = None

    def get_execution_environment(self):
        if False:
            i = 10
            return i + 15
        return self._keyed_stream.get_execution_environment()

    def get_input_type(self):
        if False:
            while True:
                i = 10
        return _from_java_type(self._keyed_stream._original_data_type_info.get_java_type_info())

    def trigger(self, trigger: Trigger) -> 'WindowedStream':
        if False:
            i = 10
            return i + 15
        '\n        Sets the Trigger that should be used to trigger window emission.\n        '
        self._window_trigger = trigger
        return self

    def allowed_lateness(self, time_ms: int) -> 'WindowedStream':
        if False:
            print('Hello World!')
        '\n        Sets the time by which elements are allowed to be late. Elements that arrive behind the\n        watermark by more than the specified time will be dropped. By default, the allowed lateness\n        is 0.\n\n        Setting an allowed lateness is only valid for event-time windows.\n        '
        self._allowed_lateness = time_ms
        return self

    def side_output_late_data(self, output_tag: OutputTag) -> 'WindowedStream':
        if False:
            return 10
        '\n        Send late arriving data to the side output identified by the given :class:`OutputTag`. Data\n        is considered late after the watermark has passed the end of the window plus the allowed\n        lateness set using :func:`allowed_lateness`.\n\n        You can get the stream of late data using :func:`~DataStream.get_side_output` on the\n        :class:`DataStream` resulting from the windowed operation with the same :class:`OutputTag`.\n\n        Example:\n        ::\n\n            >>> tag = OutputTag("late-data", Types.TUPLE([Types.INT(), Types.STRING()]))\n            >>> main_stream = ds.key_by(lambda x: x[1]) \\\n            ...                 .window(TumblingEventTimeWindows.of(Time.seconds(5))) \\\n            ...                 .side_output_late_data(tag) \\\n            ...                 .reduce(lambda a, b: a[0] + b[0], b[1])\n            >>> late_stream = main_stream.get_side_output(tag)\n\n        .. versionadded:: 1.16.0\n        '
        self._late_data_output_tag = output_tag
        return self

    def reduce(self, reduce_function: Union[Callable, ReduceFunction], window_function: Union[WindowFunction, ProcessWindowFunction]=None, output_type: TypeInformation=None) -> DataStream:
        if False:
            while True:
                i = 10
        '\n        Applies a reduce function to the window. The window function is called for each evaluation\n        of the window for each key individually. The output of the reduce function is interpreted as\n        a regular non-windowed stream.\n\n        This window will try and incrementally aggregate data as much as the window policies\n        permit. For example, tumbling time windows can aggregate the data, meaning that only one\n        element per key is stored. Sliding time windows will aggregate on the granularity of the\n        slide interval, so a few elements are stored per key (one per slide interval). Custom\n        windows may not be able to incrementally aggregate, or may need to store extra values in an\n        aggregation tree.\n\n        Example:\n        ::\n\n            >>> ds.key_by(lambda x: x[1]) \\\n            ...     .window(TumblingEventTimeWindows.of(Time.seconds(5))) \\\n            ...     .reduce(lambda a, b: a[0] + b[0], b[1])\n\n        :param reduce_function: The reduce function.\n        :param window_function: The window function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the reduce function to the window.\n\n        .. versionadded:: 1.16.0\n        '
        if window_function is None:
            internal_window_function = InternalSingleValueWindowFunction(PassThroughWindowFunction())
            if output_type is None:
                output_type = self.get_input_type()
        elif isinstance(window_function, WindowFunction):
            internal_window_function = InternalSingleValueWindowFunction(window_function)
        elif isinstance(window_function, ProcessWindowFunction):
            internal_window_function = InternalSingleValueProcessWindowFunction(window_function)
        else:
            raise TypeError('window_function should be a WindowFunction or ProcessWindowFunction')
        reducing_state_descriptor = ReducingStateDescriptor(WINDOW_STATE_NAME, reduce_function, self.get_input_type())
        func_desc = type(reduce_function).__name__
        if window_function is not None:
            func_desc = '%s, %s' % (func_desc, type(window_function).__name__)
        return self._get_result_data_stream(internal_window_function, reducing_state_descriptor, func_desc, output_type)

    def aggregate(self, aggregate_function: AggregateFunction, window_function: Union[WindowFunction, ProcessWindowFunction]=None, accumulator_type: TypeInformation=None, output_type: TypeInformation=None) -> DataStream:
        if False:
            return 10
        '\n        Applies the given window function to each window. The window function is called for each\n        evaluation of the window for each key individually. The output of the window function is\n        interpreted as a regular non-windowed stream.\n\n        Arriving data is incrementally aggregated using the given aggregate function. This means\n        that the window function typically has only a single value to process when called.\n\n        Example:\n        ::\n\n            >>> class AverageAggregate(AggregateFunction):\n            ...     def create_accumulator(self) -> Tuple[int, int]:\n            ...         return 0, 0\n            ...\n            ...     def add(self, value: Tuple[str, int], accumulator: Tuple[int, int]) \\\n            ...             -> Tuple[int, int]:\n            ...         return accumulator[0] + value[1], accumulator[1] + 1\n            ...\n            ...     def get_result(self, accumulator: Tuple[int, int]) -> float:\n            ...         return accumulator[0] / accumulator[1]\n            ...\n            ...     def merge(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:\n            ...         return a[0] + b[0], a[1] + b[1]\n            >>> ds.key_by(lambda x: x[1]) \\\n            ...     .window(TumblingEventTimeWindows.of(Time.seconds(5))) \\\n            ...     .aggregate(AverageAggregate(),\n            ...                accumulator_type=Types.TUPLE([Types.LONG(), Types.LONG()]),\n            ...                output_type=Types.DOUBLE())\n\n        :param aggregate_function: The aggregation function that is used for incremental\n                                   aggregation.\n        :param window_function: The window function.\n        :param accumulator_type: Type information for the internal accumulator type of the\n                                 aggregation function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the window function to the window.\n\n        .. versionadded:: 1.16.0\n        '
        if window_function is None:
            internal_window_function = InternalSingleValueWindowFunction(PassThroughWindowFunction())
        elif isinstance(window_function, WindowFunction):
            internal_window_function = InternalSingleValueWindowFunction(window_function)
        elif isinstance(window_function, ProcessWindowFunction):
            internal_window_function = InternalSingleValueProcessWindowFunction(window_function)
        else:
            raise TypeError('window_function should be a WindowFunction or ProcessWindowFunction')
        if accumulator_type is None:
            accumulator_type = Types.PICKLED_BYTE_ARRAY()
        elif isinstance(accumulator_type, list):
            accumulator_type = RowTypeInfo(accumulator_type)
        aggregating_state_descriptor = AggregatingStateDescriptor(WINDOW_STATE_NAME, aggregate_function, accumulator_type)
        func_desc = type(aggregate_function).__name__
        if window_function is not None:
            func_desc = '%s, %s' % (func_desc, type(window_function).__name__)
        return self._get_result_data_stream(internal_window_function, aggregating_state_descriptor, func_desc, output_type)

    def apply(self, window_function: WindowFunction, output_type: TypeInformation=None) -> DataStream:
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies the given window function to each window. The window function is called for each\n        evaluation of the window for each key individually. The output of the window function is\n        interpreted as a regular non-windowed stream.\n\n        Note that this function requires that all data in the windows is buffered until the window\n        is evaluated, as the function provides no means of incremental aggregation.\n\n        :param window_function: The window function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the window function to the window.\n        '
        internal_window_function = InternalIterableWindowFunction(window_function)
        list_state_descriptor = ListStateDescriptor(WINDOW_STATE_NAME, self.get_input_type())
        func_desc = type(window_function).__name__
        return self._get_result_data_stream(internal_window_function, list_state_descriptor, func_desc, output_type)

    def process(self, process_window_function: ProcessWindowFunction, output_type: TypeInformation=None) -> DataStream:
        if False:
            return 10
        '\n        Applies the given window function to each window. The window function is called for each\n        evaluation of the window for each key individually. The output of the window function is\n        interpreted as a regular non-windowed stream.\n\n        Note that this function requires that all data in the windows is buffered until the window\n        is evaluated, as the function provides no means of incremental aggregation.\n\n        :param process_window_function: The window function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the window function to the window.\n        '
        internal_window_function = InternalIterableProcessWindowFunction(process_window_function)
        list_state_descriptor = ListStateDescriptor(WINDOW_STATE_NAME, self.get_input_type())
        func_desc = type(process_window_function).__name__
        return self._get_result_data_stream(internal_window_function, list_state_descriptor, func_desc, output_type)

    def _get_result_data_stream(self, internal_window_function: InternalWindowFunction, window_state_descriptor: StateDescriptor, func_desc: str, output_type: TypeInformation):
        if False:
            i = 10
            return i + 15
        if self._window_trigger is None:
            self._window_trigger = self._window_assigner.get_default_trigger(self.get_execution_environment())
        window_serializer = self._window_assigner.get_window_serializer()
        window_operation_descriptor = WindowOperationDescriptor(self._window_assigner, self._window_trigger, self._allowed_lateness, self._late_data_output_tag, window_state_descriptor, window_serializer, internal_window_function)
        from pyflink.fn_execution import flink_fn_execution_pb2
        (j_python_data_stream_function_operator, j_output_type_info) = _get_one_input_stream_operator(self._keyed_stream, window_operation_descriptor, flink_fn_execution_pb2.UserDefinedDataStreamFunction.WINDOW, output_type)
        op_name = window_operation_descriptor.generate_op_name()
        op_desc = window_operation_descriptor.generate_op_desc('Window', func_desc)
        return DataStream(self._keyed_stream._j_data_stream.transform(op_name, j_output_type_info, j_python_data_stream_function_operator)).set_description(op_desc)

class AllWindowedStream(object):
    """
    A AllWindowedStream represents a data stream where the stream of elements is split into windows
    based on a WindowAssigner. Window emission is triggered based on a Trigger.

    If an Evictor is specified it will be used to evict elements from the window after evaluation
    was triggered by the Trigger but before the actual evaluation of the window.
    When using an evictor, window performance will degrade significantly, since pre-aggregation of
    window results cannot be used.

    Note that the AllWindowedStream is purely an API construct, during runtime the AllWindowedStream
    will be collapsed together with the operation over the window into one single operation.
    """

    def __init__(self, data_stream: DataStream, window_assigner: WindowAssigner):
        if False:
            print('Hello World!')
        self._keyed_stream = data_stream.key_by(NullByteKeySelector())
        self._window_assigner = window_assigner
        self._allowed_lateness = 0
        self._late_data_output_tag = None
        self._window_trigger = None

    def get_execution_environment(self):
        if False:
            while True:
                i = 10
        return self._keyed_stream.get_execution_environment()

    def get_input_type(self):
        if False:
            i = 10
            return i + 15
        return _from_java_type(self._keyed_stream._original_data_type_info.get_java_type_info())

    def trigger(self, trigger: Trigger) -> 'AllWindowedStream':
        if False:
            print('Hello World!')
        '\n        Sets the Trigger that should be used to trigger window emission.\n        '
        if isinstance(self._window_assigner, MergingWindowAssigner) and trigger.can_merge() is not True:
            raise TypeError('A merging window assigner cannot be used with a trigger that does not support merging.')
        self._window_trigger = trigger
        return self

    def allowed_lateness(self, time_ms: int) -> 'AllWindowedStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the time by which elements are allowed to be late. Elements that arrive behind the\n        watermark by more than the specified time will be dropped. By default, the allowed lateness\n        is 0.\n\n        Setting an allowed lateness is only valid for event-time windows.\n        '
        self._allowed_lateness = time_ms
        return self

    def side_output_late_data(self, output_tag: OutputTag) -> 'AllWindowedStream':
        if False:
            print('Hello World!')
        '\n        Send late arriving data to the side output identified by the given :class:`OutputTag`. Data\n        is considered late after the watermark has passed the end of the window plus the allowed\n        lateness set using :func:`allowed_lateness`.\n\n        You can get the stream of late data using :func:`~DataStream.get_side_output` on the\n        :class:`DataStream` resulting from the windowed operation with the same :class:`OutputTag`.\n\n        Example:\n        ::\n\n            >>> tag = OutputTag("late-data", Types.TUPLE([Types.INT(), Types.STRING()]))\n            >>> main_stream = ds.window_all(TumblingEventTimeWindows.of(Time.seconds(5))) \\\n            ...                 .side_output_late_data(tag) \\\n            ...                 .process(MyProcessAllWindowFunction(),\n            ...                          Types.TUPLE([Types.LONG(), Types.LONG(), Types.INT()]))\n            >>> late_stream = main_stream.get_side_output(tag)\n        '
        self._late_data_output_tag = output_tag
        return self

    def reduce(self, reduce_function: Union[Callable, ReduceFunction], window_function: Union[AllWindowFunction, ProcessAllWindowFunction]=None, output_type: TypeInformation=None) -> DataStream:
        if False:
            i = 10
            return i + 15
        '\n        Applies the given window function to each window. The window function is called for each\n        evaluation of the window for each key individually. The output of the window function is\n        interpreted as a regular non-windowed stream.\n\n        Arriving data is incrementally aggregated using the given reducer.\n\n        Example:\n        ::\n\n            >>> ds.window_all(TumblingEventTimeWindows.of(Time.seconds(5))) \\\n            ...   .reduce(lambda a, b: a[0] + b[0], b[1])\n\n        :param reduce_function: The reduce function.\n        :param window_function: The window function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the reduce function to the window.\n\n        .. versionadded:: 1.16.0\n        '
        if window_function is None:
            internal_window_function = InternalSingleValueAllWindowFunction(PassThroughAllWindowFunction())
            if output_type is None:
                output_type = self.get_input_type()
        elif isinstance(window_function, AllWindowFunction):
            internal_window_function = InternalSingleValueAllWindowFunction(window_function)
        elif isinstance(window_function, ProcessAllWindowFunction):
            internal_window_function = InternalSingleValueProcessAllWindowFunction(window_function)
        else:
            raise TypeError('window_function should be a AllWindowFunction or ProcessAllWindowFunction')
        reducing_state_descriptor = ReducingStateDescriptor(WINDOW_STATE_NAME, reduce_function, self.get_input_type())
        func_desc = type(reduce_function).__name__
        if window_function is not None:
            func_desc = '%s, %s' % (func_desc, type(window_function).__name__)
        return self._get_result_data_stream(internal_window_function, reducing_state_descriptor, func_desc, output_type)

    def aggregate(self, aggregate_function: AggregateFunction, window_function: Union[AllWindowFunction, ProcessAllWindowFunction]=None, accumulator_type: TypeInformation=None, output_type: TypeInformation=None) -> DataStream:
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies the given window function to each window. The window function is called for each\n        evaluation of the window for each key individually. The output of the window function is\n        interpreted as a regular non-windowed stream.\n\n        Arriving data is incrementally aggregated using the given aggregate function. This means\n        that the window function typically has only a single value to process when called.\n\n        Example:\n        ::\n\n            >>> class AverageAggregate(AggregateFunction):\n            ...     def create_accumulator(self) -> Tuple[int, int]:\n            ...         return 0, 0\n            ...\n            ...     def add(self, value: Tuple[str, int], accumulator: Tuple[int, int]) \\\n            ...             -> Tuple[int, int]:\n            ...         return accumulator[0] + value[1], accumulator[1] + 1\n            ...\n            ...     def get_result(self, accumulator: Tuple[int, int]) -> float:\n            ...         return accumulator[0] / accumulator[1]\n            ...\n            ...     def merge(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:\n            ...         return a[0] + b[0], a[1] + b[1]\n            ...\n            >>> ds.window_all(TumblingEventTimeWindows.of(Time.seconds(5))) \\\n            ...   .aggregate(AverageAggregate(),\n            ...              accumulator_type=Types.TUPLE([Types.LONG(), Types.LONG()]),\n            ...              output_type=Types.DOUBLE())\n\n        :param aggregate_function: The aggregation function that is used for incremental\n                                   aggregation.\n        :param window_function: The window function.\n        :param accumulator_type: Type information for the internal accumulator type of the\n                                 aggregation function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the window function to the window.\n\n        .. versionadded:: 1.16.0\n        '
        if window_function is None:
            internal_window_function = InternalSingleValueAllWindowFunction(PassThroughAllWindowFunction())
        elif isinstance(window_function, AllWindowFunction):
            internal_window_function = InternalSingleValueAllWindowFunction(window_function)
        elif isinstance(window_function, ProcessAllWindowFunction):
            internal_window_function = InternalSingleValueProcessAllWindowFunction(window_function)
        else:
            raise TypeError('window_function should be a AllWindowFunction or ProcessAllWindowFunction')
        if accumulator_type is None:
            accumulator_type = Types.PICKLED_BYTE_ARRAY()
        elif isinstance(accumulator_type, list):
            accumulator_type = RowTypeInfo(accumulator_type)
        aggregating_state_descriptor = AggregatingStateDescriptor(WINDOW_STATE_NAME, aggregate_function, accumulator_type)
        func_desc = type(aggregate_function).__name__
        if window_function is not None:
            func_desc = '%s, %s' % (func_desc, type(window_function).__name__)
        return self._get_result_data_stream(internal_window_function, aggregating_state_descriptor, func_desc, output_type)

    def apply(self, window_function: AllWindowFunction, output_type: TypeInformation=None) -> DataStream:
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies the given window function to each window. The window function is called for each\n        evaluation of the window. The output of the window function is interpreted as a regular\n        non-windowed stream.\n\n        Note that this function requires that all data in the windows is buffered until the window\n        is evaluated, as the function provides no means of incremental aggregation.\n\n        :param window_function: The window function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the window function to the window.\n        '
        internal_window_function = InternalIterableAllWindowFunction(window_function)
        list_state_descriptor = ListStateDescriptor(WINDOW_STATE_NAME, self.get_input_type())
        func_desc = type(window_function).__name__
        return self._get_result_data_stream(internal_window_function, list_state_descriptor, func_desc, output_type)

    def process(self, process_window_function: ProcessAllWindowFunction, output_type: TypeInformation=None) -> DataStream:
        if False:
            while True:
                i = 10
        '\n        Applies the given window function to each window. The window function is called for each\n        evaluation of the window for each key individually. The output of the window function is\n        interpreted as a regular non-windowed stream.\n\n        Note that this function requires that all data in the windows is buffered until the window\n        is evaluated, as the function provides no means of incremental aggregation.\n\n        :param process_window_function: The window function.\n        :param output_type: Type information for the result type of the window function.\n        :return: The data stream that is the result of applying the window function to the window.\n        '
        internal_window_function = InternalIterableProcessAllWindowFunction(process_window_function)
        list_state_descriptor = ListStateDescriptor(WINDOW_STATE_NAME, self.get_input_type())
        func_desc = type(process_window_function).__name__
        return self._get_result_data_stream(internal_window_function, list_state_descriptor, func_desc, output_type)

    def _get_result_data_stream(self, internal_window_function: InternalWindowFunction, window_state_descriptor: StateDescriptor, func_desc: str, output_type: TypeInformation):
        if False:
            for i in range(10):
                print('nop')
        if self._window_trigger is None:
            self._window_trigger = self._window_assigner.get_default_trigger(self.get_execution_environment())
        window_serializer = self._window_assigner.get_window_serializer()
        window_operation_descriptor = WindowOperationDescriptor(self._window_assigner, self._window_trigger, self._allowed_lateness, self._late_data_output_tag, window_state_descriptor, window_serializer, internal_window_function)
        from pyflink.fn_execution import flink_fn_execution_pb2
        (j_python_data_stream_function_operator, j_output_type_info) = _get_one_input_stream_operator(self._keyed_stream, window_operation_descriptor, flink_fn_execution_pb2.UserDefinedDataStreamFunction.WINDOW, output_type)
        op_name = window_operation_descriptor.generate_op_name()
        op_desc = window_operation_descriptor.generate_op_desc('AllWindow', func_desc)
        return DataStream(self._keyed_stream._j_data_stream.transform(op_name, j_output_type_info, j_python_data_stream_function_operator)).set_description(op_desc)

class ConnectedStreams(object):
    """
    ConnectedStreams represent two connected streams of (possibly) different data types.
    Connected streams are useful for cases where operations on one stream directly
    affect the operations on the other stream, usually via shared state between the streams.

    An example for the use of connected streams would be to apply rules that change over time
    onto another stream. One of the connected streams has the rules, the other stream the
    elements to apply the rules to. The operation on the connected stream maintains the
    current set of rules in the state. It may receive either a rule update and update the state
    or a data element and apply the rules in the state to the element.

    The connected stream can be conceptually viewed as a union stream of an Either type, that
    holds either the first stream's type or the second stream's type.
    """

    def __init__(self, stream1: DataStream, stream2: DataStream):
        if False:
            for i in range(10):
                print('nop')
        self.stream1 = stream1
        self.stream2 = stream2

    def key_by(self, key_selector1: Union[Callable, KeySelector], key_selector2: Union[Callable, KeySelector], key_type: TypeInformation=None) -> 'ConnectedStreams':
        if False:
            for i in range(10):
                print('nop')
        '\n        KeyBy operation for connected data stream. Assigns keys to the elements of\n        input1 and input2 using keySelector1 and keySelector2 with explicit type information\n        for the common key type.\n\n        :param key_selector1: The `KeySelector` used for grouping the first input.\n        :param key_selector2: The `KeySelector` used for grouping the second input.\n        :param key_type: The type information of the common key type\n        :return: The partitioned `ConnectedStreams`\n        '
        ds1 = self.stream1
        ds2 = self.stream2
        if isinstance(self.stream1, KeyedStream):
            ds1 = self.stream1._origin_stream
        if isinstance(self.stream2, KeyedStream):
            ds2 = self.stream2._origin_stream
        return ConnectedStreams(ds1.key_by(key_selector1, key_type), ds2.key_by(key_selector2, key_type))

    def map(self, func: CoMapFunction, output_type: TypeInformation=None) -> 'DataStream':
        if False:
            print('Hello World!')
        '\n        Applies a CoMap transformation on a `ConnectedStreams` and maps the output to a common\n        type. The transformation calls a `CoMapFunction.map1` for each element of the first\n        input and `CoMapFunction.map2` for each element of the second input. Each CoMapFunction\n        call returns exactly one element.\n\n        :param func: The CoMapFunction used to jointly transform the two input DataStreams\n        :param output_type: `TypeInformation` for the result type of the function.\n        :return: The transformed `DataStream`\n        '
        if not isinstance(func, CoMapFunction):
            raise TypeError('The input function must be a CoMapFunction!')
        if self._is_keyed_stream():

            class CoMapKeyedCoProcessFunctionAdapter(KeyedCoProcessFunction):

                def __init__(self, co_map_func: CoMapFunction):
                    if False:
                        while True:
                            i = 10
                    self._open_func = co_map_func.open
                    self._close_func = co_map_func.close
                    self._map1_func = co_map_func.map1
                    self._map2_func = co_map_func.map2

                def open(self, runtime_context: RuntimeContext):
                    if False:
                        return 10
                    self._open_func(runtime_context)

                def close(self):
                    if False:
                        return 10
                    self._close_func()

                def process_element1(self, value, ctx: 'KeyedCoProcessFunction.Context'):
                    if False:
                        print('Hello World!')
                    result = self._map1_func(value)
                    if result is not None:
                        yield result

                def process_element2(self, value, ctx: 'KeyedCoProcessFunction.Context'):
                    if False:
                        print('Hello World!')
                    result = self._map2_func(value)
                    if result is not None:
                        yield result
            return self.process(CoMapKeyedCoProcessFunctionAdapter(func), output_type).name('Co-Map')
        else:

            class CoMapCoProcessFunctionAdapter(CoProcessFunction):

                def __init__(self, co_map_func: CoMapFunction):
                    if False:
                        while True:
                            i = 10
                    self._open_func = co_map_func.open
                    self._close_func = co_map_func.close
                    self._map1_func = co_map_func.map1
                    self._map2_func = co_map_func.map2

                def open(self, runtime_context: RuntimeContext):
                    if False:
                        for i in range(10):
                            print('nop')
                    self._open_func(runtime_context)

                def close(self):
                    if False:
                        while True:
                            i = 10
                    self._close_func()

                def process_element1(self, value, ctx: 'CoProcessFunction.Context'):
                    if False:
                        print('Hello World!')
                    result = self._map1_func(value)
                    if result is not None:
                        yield result

                def process_element2(self, value, ctx: 'CoProcessFunction.Context'):
                    if False:
                        i = 10
                        return i + 15
                    result = self._map2_func(value)
                    if result is not None:
                        yield result
            return self.process(CoMapCoProcessFunctionAdapter(func), output_type).name('Co-Map')

    def flat_map(self, func: CoFlatMapFunction, output_type: TypeInformation=None) -> 'DataStream':
        if False:
            return 10
        '\n        Applies a CoFlatMap transformation on a `ConnectedStreams` and maps the output to a\n        common type. The transformation calls a `CoFlatMapFunction.flatMap1` for each element\n        of the first input and `CoFlatMapFunction.flatMap2` for each element of the second\n        input. Each CoFlatMapFunction call returns any number of elements including none.\n\n        :param func: The CoFlatMapFunction used to jointly transform the two input DataStreams\n        :param output_type: `TypeInformation` for the result type of the function.\n        :return: The transformed `DataStream`\n        '
        if not isinstance(func, CoFlatMapFunction):
            raise TypeError('The input must be a CoFlatMapFunction!')
        if self._is_keyed_stream():

            class FlatMapKeyedCoProcessFunctionAdapter(KeyedCoProcessFunction):

                def __init__(self, co_flat_map_func: CoFlatMapFunction):
                    if False:
                        for i in range(10):
                            print('nop')
                    self._open_func = co_flat_map_func.open
                    self._close_func = co_flat_map_func.close
                    self._flat_map1_func = co_flat_map_func.flat_map1
                    self._flat_map2_func = co_flat_map_func.flat_map2

                def open(self, runtime_context: RuntimeContext):
                    if False:
                        i = 10
                        return i + 15
                    self._open_func(runtime_context)

                def close(self):
                    if False:
                        print('Hello World!')
                    self._close_func()

                def process_element1(self, value, ctx: 'KeyedCoProcessFunction.Context'):
                    if False:
                        for i in range(10):
                            print('nop')
                    result = self._flat_map1_func(value)
                    if result:
                        yield from result

                def process_element2(self, value, ctx: 'KeyedCoProcessFunction.Context'):
                    if False:
                        i = 10
                        return i + 15
                    result = self._flat_map2_func(value)
                    if result:
                        yield from result
            return self.process(FlatMapKeyedCoProcessFunctionAdapter(func), output_type).name('Co-Flat Map')
        else:

            class FlatMapCoProcessFunctionAdapter(CoProcessFunction):

                def __init__(self, co_flat_map_func: CoFlatMapFunction):
                    if False:
                        while True:
                            i = 10
                    self._open_func = co_flat_map_func.open
                    self._close_func = co_flat_map_func.close
                    self._flat_map1_func = co_flat_map_func.flat_map1
                    self._flat_map2_func = co_flat_map_func.flat_map2

                def open(self, runtime_context: RuntimeContext):
                    if False:
                        print('Hello World!')
                    self._open_func(runtime_context)

                def close(self):
                    if False:
                        i = 10
                        return i + 15
                    self._close_func()

                def process_element1(self, value, ctx: 'CoProcessFunction.Context'):
                    if False:
                        for i in range(10):
                            print('nop')
                    result = self._flat_map1_func(value)
                    if result:
                        yield from result

                def process_element2(self, value, ctx: 'CoProcessFunction.Context'):
                    if False:
                        print('Hello World!')
                    result = self._flat_map2_func(value)
                    if result:
                        yield from result
            return self.process(FlatMapCoProcessFunctionAdapter(func), output_type).name('Co-Flat Map')

    def process(self, func: Union[CoProcessFunction, KeyedCoProcessFunction], output_type: TypeInformation=None) -> 'DataStream':
        if False:
            print('Hello World!')
        if not isinstance(func, CoProcessFunction) and (not isinstance(func, KeyedCoProcessFunction)):
            raise TypeError('The input must be a CoProcessFunction or KeyedCoProcessFunction!')
        from pyflink.fn_execution.flink_fn_execution_pb2 import UserDefinedDataStreamFunction
        if self._is_keyed_stream():
            func_type = UserDefinedDataStreamFunction.KEYED_CO_PROCESS
            func_name = 'Keyed Co-Process'
        else:
            func_type = UserDefinedDataStreamFunction.CO_PROCESS
            func_name = 'Co-Process'
        j_connected_stream = self.stream1._j_data_stream.connect(self.stream2._j_data_stream)
        (j_operator, j_output_type) = _get_two_input_stream_operator(self, func, func_type, output_type)
        return DataStream(j_connected_stream.transform(func_name, j_output_type, j_operator))

    def _is_keyed_stream(self):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(self.stream1, KeyedStream) and isinstance(self.stream2, KeyedStream)

class BroadcastStream(object):
    """
    A BroadcastStream is a stream with :class:`state.BroadcastState` (s). This can be created by any
    stream using the :meth:`DataStream.broadcast` method and implicitly creates states where the
    user can store elements of the created :class:`BroadcastStream`. (see
    :class:`BroadcastConnectedStream`).

    Note that no further operation can be applied to these streams. The only available option is
    to connect them with a keyed or non-keyed stream, using the :meth:`KeyedStream.connect` and the
    :meth:`DataStream.connect` respectively. Applying these methods will result it a
    :class:`BroadcastConnectedStream` for further processing.

    .. versionadded:: 1.16.0
    """

    def __init__(self, input_stream: Union['DataStream', 'KeyedStream'], broadcast_state_descriptors: List[MapStateDescriptor]):
        if False:
            return 10
        self.input_stream = input_stream
        self.broadcast_state_descriptors = broadcast_state_descriptors

class BroadcastConnectedStream(object):
    """
    A BroadcastConnectedStream represents the result of connecting a keyed or non-keyed stream, with
    a :class:`BroadcastStream` with :class:`~state.BroadcastState` (s). As in the case of
    :class:`ConnectedStreams` these streams are useful for cases where operations on one stream
    directly affect the operations on the other stream, usually via shared state between the
    streams.

    An example for the use of such connected streams would be to apply rules that change over time
    onto another, possibly keyed stream. The stream with the broadcast state has the rules, and will
    store them in the broadcast state, while the other stream will contain the elements to apply the
    rules to. By broadcasting the rules, these will be available in all parallel instances, and can
    be applied to all partitions of the other stream.

    .. versionadded:: 1.16.0
    """

    def __init__(self, non_broadcast_stream: Union['DataStream', 'KeyedStream'], broadcast_stream: 'BroadcastStream', broadcast_state_descriptors: List[MapStateDescriptor]):
        if False:
            for i in range(10):
                print('nop')
        self.non_broadcast_stream = non_broadcast_stream
        self.broadcast_stream = broadcast_stream
        self.broadcast_state_descriptors = broadcast_state_descriptors

    @overload
    def process(self, func: BroadcastProcessFunction, output_type: TypeInformation=None) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        pass

    @overload
    def process(self, func: KeyedBroadcastProcessFunction, output_type: TypeInformation=None) -> 'DataStream':
        if False:
            i = 10
            return i + 15
        pass

    def process(self, func: Union[BroadcastProcessFunction, KeyedBroadcastProcessFunction], output_type: TypeInformation=None) -> 'DataStream':
        if False:
            return 10
        '\n        Assumes as inputs a :class:`BroadcastStream` and a :class:`DataStream` or\n        :class:`KeyedStream` and applies the given :class:`BroadcastProcessFunction` or\n        :class:`KeyedBroadcastProcessFunction` on them, thereby creating a transformed output\n        stream.\n\n        :param func: The :class:`BroadcastProcessFunction` that is called for each element in the\n            non-broadcasted :class:`DataStream`, or the :class:`KeyedBroadcastProcessFunction` that\n            is called for each element in the non-broadcasted :class:`KeyedStream`.\n        :param output_type: The type of the output elements, should be\n            :class:`common.TypeInformation` or list (implicit :class:`RowTypeInfo`) or None (\n            implicit :meth:`Types.PICKLED_BYTE_ARRAY`).\n        :return: The transformed :class:`DataStream`.\n        '
        if isinstance(func, BroadcastProcessFunction) and self._is_keyed_stream():
            raise TypeError('BroadcastProcessFunction should be applied to non-keyed DataStream')
        if isinstance(func, KeyedBroadcastProcessFunction) and (not self._is_keyed_stream()):
            raise TypeError('KeyedBroadcastProcessFunction should be applied to keyed DataStream')
        j_input_transformation1 = self.non_broadcast_stream._j_data_stream.getTransformation()
        j_input_transformation2 = self.broadcast_stream.input_stream._j_data_stream.getTransformation()
        if output_type is None:
            output_type_info = Types.PICKLED_BYTE_ARRAY()
        elif isinstance(output_type, list):
            output_type_info = RowTypeInfo(output_type)
        elif isinstance(output_type, TypeInformation):
            output_type_info = output_type
        else:
            raise TypeError('output_type must be None, list or TypeInformation')
        j_output_type = output_type_info.get_java_type_info()
        from pyflink.fn_execution.flink_fn_execution_pb2 import UserDefinedDataStreamFunction
        jvm = get_gateway().jvm
        JPythonConfigUtil = jvm.org.apache.flink.python.util.PythonConfigUtil
        if self._is_keyed_stream():
            func_type = UserDefinedDataStreamFunction.KEYED_CO_BROADCAST_PROCESS
            func_name = 'Keyed-Co-Process-Broadcast'
        else:
            func_type = UserDefinedDataStreamFunction.CO_BROADCAST_PROCESS
            func_name = 'Co-Process-Broadcast'
        j_state_names = to_jarray(jvm.String, [i.get_name() for i in self.broadcast_state_descriptors])
        j_state_descriptors = JPythonConfigUtil.convertStateNamesToStateDescriptors(j_state_names)
        j_conf = get_j_env_configuration(self.broadcast_stream.input_stream._j_data_stream.getExecutionEnvironment())
        j_data_stream_python_function_info = _create_j_data_stream_python_function_info(func, func_type)
        j_env = self.non_broadcast_stream.get_execution_environment()._j_stream_execution_environment
        if self._is_keyed_stream():
            JTransformation = jvm.org.apache.flink.streaming.api.transformations.python.PythonKeyedBroadcastStateTransformation
            j_transformation = JTransformation(func_name, j_conf, j_data_stream_python_function_info, j_input_transformation1, j_input_transformation2, j_state_descriptors, self.non_broadcast_stream._j_data_stream.getKeyType(), self.non_broadcast_stream._j_data_stream.getKeySelector(), j_output_type, j_env.getParallelism())
        else:
            JTransformation = jvm.org.apache.flink.streaming.api.transformations.python.PythonBroadcastStateTransformation
            j_transformation = JTransformation(func_name, j_conf, j_data_stream_python_function_info, j_input_transformation1, j_input_transformation2, j_state_descriptors, j_output_type, j_env.getParallelism())
        j_env.addOperator(j_transformation)
        j_data_stream = JPythonConfigUtil.createSingleOutputStreamOperator(j_env, j_transformation)
        return DataStream(j_data_stream)

    def _is_keyed_stream(self):
        if False:
            while True:
                i = 10
        return isinstance(self.non_broadcast_stream, KeyedStream)

def _get_one_input_stream_operator(data_stream: DataStream, func: Union[Function, FunctionWrapper, WindowOperationDescriptor], func_type: int, output_type: Union[TypeInformation, List]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Java one input stream operator.\n\n    :param func: a function object that implements the Function interface.\n    :param func_type: function type, supports MAP, FLAT_MAP, etc.\n    :param output_type: the data type of the function output data.\n    :return: A Java operator which is responsible for execution user defined python function.\n    '
    gateway = get_gateway()
    j_input_types = data_stream._j_data_stream.getTransformation().getOutputType()
    if output_type is None:
        output_type_info = Types.PICKLED_BYTE_ARRAY()
    elif isinstance(output_type, list):
        output_type_info = RowTypeInfo(output_type)
    else:
        output_type_info = output_type
    j_data_stream_python_function_info = _create_j_data_stream_python_function_info(func, func_type)
    j_output_type_info = output_type_info.get_java_type_info()
    j_conf = get_j_env_configuration(data_stream._j_data_stream.getExecutionEnvironment())
    python_execution_mode = j_conf.getString(gateway.jvm.org.apache.flink.python.PythonOptions.PYTHON_EXECUTION_MODE)
    from pyflink.fn_execution.flink_fn_execution_pb2 import UserDefinedDataStreamFunction
    if func_type == UserDefinedDataStreamFunction.PROCESS:
        if python_execution_mode == 'thread':
            JDataStreamPythonFunctionOperator = gateway.jvm.EmbeddedPythonProcessOperator
        else:
            JDataStreamPythonFunctionOperator = gateway.jvm.ExternalPythonProcessOperator
    elif func_type == UserDefinedDataStreamFunction.KEYED_PROCESS:
        if python_execution_mode == 'thread':
            JDataStreamPythonFunctionOperator = gateway.jvm.EmbeddedPythonKeyedProcessOperator
        else:
            JDataStreamPythonFunctionOperator = gateway.jvm.ExternalPythonKeyedProcessOperator
    elif func_type == UserDefinedDataStreamFunction.WINDOW:
        window_serializer = typing.cast(WindowOperationDescriptor, func).window_serializer
        if isinstance(window_serializer, TimeWindowSerializer):
            j_namespace_serializer = gateway.jvm.org.apache.flink.table.runtime.operators.window.TimeWindow.Serializer()
        elif isinstance(window_serializer, CountWindowSerializer):
            j_namespace_serializer = gateway.jvm.org.apache.flink.table.runtime.operators.window.CountWindow.Serializer()
        elif isinstance(window_serializer, GlobalWindowSerializer):
            j_namespace_serializer = gateway.jvm.org.apache.flink.streaming.api.windowing.windows.GlobalWindow.Serializer()
        else:
            j_namespace_serializer = gateway.jvm.org.apache.flink.streaming.api.utils.ByteArrayWrapperSerializer()
        if python_execution_mode == 'thread':
            JDataStreamPythonWindowFunctionOperator = gateway.jvm.EmbeddedPythonWindowOperator
        else:
            JDataStreamPythonWindowFunctionOperator = gateway.jvm.ExternalPythonKeyedProcessOperator
        j_python_function_operator = JDataStreamPythonWindowFunctionOperator(j_conf, j_data_stream_python_function_info, j_input_types, j_output_type_info, j_namespace_serializer)
        return (j_python_function_operator, j_output_type_info)
    else:
        raise TypeError('Unsupported function type: %s' % func_type)
    j_python_function_operator = JDataStreamPythonFunctionOperator(j_conf, j_data_stream_python_function_info, j_input_types, j_output_type_info)
    return (j_python_function_operator, j_output_type_info)

def _get_two_input_stream_operator(connected_streams: ConnectedStreams, func: Union[Function, FunctionWrapper], func_type: int, type_info: TypeInformation):
    if False:
        return 10
    '\n    Create a Java two input stream operator.\n\n    :param func: a function object that implements the Function interface.\n    :param func_type: function type, supports MAP, FLAT_MAP, etc.\n    :param type_info: the data type of the function output data.\n    :return: A Java operator which is responsible for execution user defined python function.\n    '
    gateway = get_gateway()
    j_input_types1 = connected_streams.stream1._j_data_stream.getTransformation().getOutputType()
    j_input_types2 = connected_streams.stream2._j_data_stream.getTransformation().getOutputType()
    if type_info is None:
        output_type_info = Types.PICKLED_BYTE_ARRAY()
    elif isinstance(type_info, list):
        output_type_info = RowTypeInfo(type_info)
    else:
        output_type_info = type_info
    j_data_stream_python_function_info = _create_j_data_stream_python_function_info(func, func_type)
    j_output_type_info = output_type_info.get_java_type_info()
    j_conf = get_j_env_configuration(connected_streams.stream1._j_data_stream.getExecutionEnvironment())
    python_execution_mode = j_conf.getString(gateway.jvm.org.apache.flink.python.PythonOptions.PYTHON_EXECUTION_MODE)
    from pyflink.fn_execution.flink_fn_execution_pb2 import UserDefinedDataStreamFunction
    if func_type == UserDefinedDataStreamFunction.CO_PROCESS:
        if python_execution_mode == 'thread':
            JTwoInputPythonFunctionOperator = gateway.jvm.EmbeddedPythonCoProcessOperator
        else:
            JTwoInputPythonFunctionOperator = gateway.jvm.ExternalPythonCoProcessOperator
    elif func_type == UserDefinedDataStreamFunction.KEYED_CO_PROCESS:
        if python_execution_mode == 'thread':
            JTwoInputPythonFunctionOperator = gateway.jvm.EmbeddedPythonKeyedCoProcessOperator
        else:
            JTwoInputPythonFunctionOperator = gateway.jvm.ExternalPythonKeyedCoProcessOperator
    else:
        raise TypeError('Unsupported function type: %s' % func_type)
    j_python_data_stream_function_operator = JTwoInputPythonFunctionOperator(j_conf, j_data_stream_python_function_info, j_input_types1, j_input_types2, j_output_type_info)
    return (j_python_data_stream_function_operator, j_output_type_info)

def _create_j_data_stream_python_function_info(func: Union[Function, FunctionWrapper, WindowOperationDescriptor], func_type: int) -> bytes:
    if False:
        print('Hello World!')
    gateway = get_gateway()
    import cloudpickle
    serialized_func = cloudpickle.dumps(func)
    j_data_stream_python_function = gateway.jvm.DataStreamPythonFunction(bytearray(serialized_func), _get_python_env())
    return gateway.jvm.DataStreamPythonFunctionInfo(j_data_stream_python_function, func_type)

class CloseableIterator(object):
    """
    Representing an Iterator that is also auto closeable.
    """

    def __init__(self, j_closeable_iterator, type_info: TypeInformation=None):
        if False:
            for i in range(10):
                print('nop')
        self._j_closeable_iterator = j_closeable_iterator
        self._type_info = type_info

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        return self.next()

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        self.close()

    def next(self):
        if False:
            i = 10
            return i + 15
        if not self._j_closeable_iterator.hasNext():
            raise StopIteration('No more data.')
        return convert_to_python_obj(self._j_closeable_iterator.next(), self._type_info)

    def close(self):
        if False:
            while True:
                i = 10
        self._j_closeable_iterator.close()