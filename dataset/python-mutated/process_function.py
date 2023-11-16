from abc import ABC
from pyflink.datastream import ProcessFunction, KeyedProcessFunction, CoProcessFunction, KeyedCoProcessFunction, TimerService, TimeDomain
from pyflink.datastream.functions import BaseBroadcastProcessFunction, BroadcastProcessFunction, KeyedBroadcastProcessFunction
from pyflink.datastream.state import MapStateDescriptor, BroadcastState, ReadOnlyBroadcastState
from pyflink.fn_execution.datastream.embedded.state_impl import ReadOnlyBroadcastStateImpl, BroadcastStateImpl
from pyflink.fn_execution.datastream.embedded.timerservice_impl import TimerServiceImpl
from pyflink.fn_execution.embedded.converters import from_type_info_proto, from_type_info
from pyflink.fn_execution.embedded.java_utils import to_java_state_descriptor

class InternalProcessFunctionContext(ProcessFunction.Context, CoProcessFunction.Context, TimerService):

    def __init__(self, j_context):
        if False:
            print('Hello World!')
        self._j_context = j_context

    def timer_service(self) -> TimerService:
        if False:
            for i in range(10):
                print('nop')
        return self

    def timestamp(self) -> int:
        if False:
            print('Hello World!')
        return self._j_context.timestamp()

    def current_processing_time(self):
        if False:
            while True:
                i = 10
        return self._j_context.currentProcessingTime()

    def current_watermark(self):
        if False:
            return 10
        return self._j_context.currentWatermark()

    def register_processing_time_timer(self, timestamp: int):
        if False:
            i = 10
            return i + 15
        raise Exception('Register timers is only supported on a keyed stream.')

    def register_event_time_timer(self, timestamp: int):
        if False:
            while True:
                i = 10
        raise Exception('Register timers is only supported on a keyed stream.')

    def delete_processing_time_timer(self, t: int):
        if False:
            i = 10
            return i + 15
        raise Exception('Deleting timers is only supported on a keyed streams.')

    def delete_event_time_timer(self, t: int):
        if False:
            return 10
        raise Exception('Deleting timers is only supported on a keyed streams.')

class InternalKeyedProcessFunctionContext(KeyedProcessFunction.Context, KeyedCoProcessFunction.Context):

    def __init__(self, j_context, key_type_info):
        if False:
            while True:
                i = 10
        self._j_context = j_context
        self._timer_service = TimerServiceImpl(self._j_context.timerService())
        self._key_converter = from_type_info_proto(key_type_info)

    def get_current_key(self):
        if False:
            i = 10
            return i + 15
        return self._key_converter.to_internal(self._j_context.getCurrentKey())

    def timer_service(self) -> TimerService:
        if False:
            while True:
                i = 10
        return self._timer_service

    def timestamp(self) -> int:
        if False:
            print('Hello World!')
        return self._j_context.timestamp()

class InternalKeyedProcessFunctionOnTimerContext(KeyedProcessFunction.OnTimerContext, KeyedProcessFunction.Context, KeyedCoProcessFunction.OnTimerContext, KeyedCoProcessFunction.Context):

    def __init__(self, j_timer_context, key_type_info):
        if False:
            while True:
                i = 10
        self._j_timer_context = j_timer_context
        self._timer_service = TimerServiceImpl(self._j_timer_context.timerService())
        self._key_converter = from_type_info_proto(key_type_info)

    def timer_service(self) -> TimerService:
        if False:
            for i in range(10):
                print('nop')
        return self._timer_service

    def timestamp(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._j_timer_context.timestamp()

    def time_domain(self) -> TimeDomain:
        if False:
            for i in range(10):
                print('nop')
        return TimeDomain(self._j_timer_context.timeDomain())

    def get_current_key(self):
        if False:
            i = 10
            return i + 15
        return self._key_converter.to_internal(self._j_timer_context.getCurrentKey())

class InternalWindowTimerContext(object):

    def __init__(self, j_timer_context, key_type_info, window_converter):
        if False:
            for i in range(10):
                print('nop')
        self._j_timer_context = j_timer_context
        self._key_converter = from_type_info_proto(key_type_info)
        self._window_converter = window_converter

    def timestamp(self) -> int:
        if False:
            return 10
        return self._j_timer_context.timestamp()

    def window(self):
        if False:
            i = 10
            return i + 15
        return self._window_converter.to_internal(self._j_timer_context.getWindow())

    def get_current_key(self):
        if False:
            for i in range(10):
                print('nop')
        return self._key_converter.to_internal(self._j_timer_context.getCurrentKey())

class InternalBaseBroadcastProcessFunctionContext(BaseBroadcastProcessFunction.Context, ABC):

    def __init__(self, j_context, j_operator_state_backend):
        if False:
            i = 10
            return i + 15
        self._j_context = j_context
        self._j_operator_state_backend = j_operator_state_backend

    def timestamp(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._j_context.timestamp()

    def current_processing_time(self) -> int:
        if False:
            while True:
                i = 10
        return self._j_context.currentProcessingTime()

    def current_watermark(self) -> int:
        if False:
            while True:
                i = 10
        return self._j_context.currentWatermark()

class InternalBroadcastProcessFunctionContext(InternalBaseBroadcastProcessFunctionContext, BroadcastProcessFunction.Context):

    def __init__(self, j_context, j_operator_state_backend):
        if False:
            for i in range(10):
                print('nop')
        super(InternalBroadcastProcessFunctionContext, self).__init__(j_context, j_operator_state_backend)

    def get_broadcast_state(self, state_descriptor: MapStateDescriptor) -> BroadcastState:
        if False:
            i = 10
            return i + 15
        return BroadcastStateImpl(self._j_operator_state_backend.getBroadcastState(to_java_state_descriptor(state_descriptor)), from_type_info(state_descriptor.type_info))

class InternalBroadcastProcessFunctionReadOnlyContext(InternalBaseBroadcastProcessFunctionContext, BroadcastProcessFunction.ReadOnlyContext):

    def __init__(self, j_context, j_operator_state_backend):
        if False:
            print('Hello World!')
        super(InternalBroadcastProcessFunctionReadOnlyContext, self).__init__(j_context, j_operator_state_backend)

    def get_broadcast_state(self, state_descriptor: MapStateDescriptor) -> ReadOnlyBroadcastState:
        if False:
            i = 10
            return i + 15
        return ReadOnlyBroadcastStateImpl(self._j_operator_state_backend.getBroadcastState(to_java_state_descriptor(state_descriptor)), from_type_info(state_descriptor.type_info))

class InternalKeyedBroadcastProcessFunctionContext(InternalBaseBroadcastProcessFunctionContext, KeyedBroadcastProcessFunction.Context):

    def __init__(self, j_context, j_operator_state_backend):
        if False:
            while True:
                i = 10
        super(InternalKeyedBroadcastProcessFunctionContext, self).__init__(j_context, j_operator_state_backend)

    def get_broadcast_state(self, state_descriptor: MapStateDescriptor) -> BroadcastState:
        if False:
            return 10
        return BroadcastStateImpl(self._j_operator_state_backend.getBroadcastState(to_java_state_descriptor(state_descriptor)), from_type_info(state_descriptor.type_info))

class InternalKeyedBroadcastProcessFunctionReadOnlyContext(InternalBaseBroadcastProcessFunctionContext, KeyedBroadcastProcessFunction.ReadOnlyContext):

    def __init__(self, j_context, key_type_info, j_operator_state_backend):
        if False:
            print('Hello World!')
        super(InternalKeyedBroadcastProcessFunctionReadOnlyContext, self).__init__(j_context, j_operator_state_backend)
        self._key_converter = from_type_info_proto(key_type_info)
        self._timer_service = TimerServiceImpl(self._j_context.timerService())

    def get_broadcast_state(self, state_descriptor: MapStateDescriptor) -> ReadOnlyBroadcastState:
        if False:
            while True:
                i = 10
        return ReadOnlyBroadcastStateImpl(self._j_operator_state_backend.getBroadcastState(to_java_state_descriptor(state_descriptor)), from_type_info(state_descriptor.type_info))

    def timer_service(self) -> TimerService:
        if False:
            while True:
                i = 10
        return self._timer_service

    def get_current_key(self):
        if False:
            for i in range(10):
                print('nop')
        return self._key_converter.to_internal(self._j_context.getCurrentKey())

class InternalKeyedBroadcastProcessFunctionOnTimerContext(InternalBaseBroadcastProcessFunctionContext, KeyedBroadcastProcessFunction.OnTimerContext):

    def __init__(self, j_timer_context, key_type_info, j_operator_state_backend):
        if False:
            while True:
                i = 10
        super(InternalKeyedBroadcastProcessFunctionOnTimerContext, self).__init__(j_timer_context, j_operator_state_backend)
        self._timer_service = TimerServiceImpl(self._j_context.timerService())
        self._key_converter = from_type_info_proto(key_type_info)

    def get_broadcast_state(self, state_descriptor: MapStateDescriptor) -> ReadOnlyBroadcastState:
        if False:
            i = 10
            return i + 15
        return ReadOnlyBroadcastStateImpl(self._j_operator_state_backend.getBroadcastState(to_java_state_descriptor(state_descriptor)), from_type_info(state_descriptor.type_info))

    def current_processing_time(self) -> int:
        if False:
            while True:
                i = 10
        return self._timer_service.current_processing_time()

    def current_watermark(self) -> int:
        if False:
            print('Hello World!')
        return self._timer_service.current_watermark()

    def timer_service(self) -> TimerService:
        if False:
            while True:
                i = 10
        return self._timer_service

    def timestamp(self) -> int:
        if False:
            while True:
                i = 10
        return self._j_context.timestamp()

    def time_domain(self) -> TimeDomain:
        if False:
            for i in range(10):
                print('nop')
        return TimeDomain(self._j_context.timeDomain())

    def get_current_key(self):
        if False:
            while True:
                i = 10
        return self._key_converter.to_internal(self._j_context.getCurrentKey())