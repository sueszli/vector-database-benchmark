from pyflink.common import Time, WatermarkStrategy, Duration
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig

class Sum(KeyedProcessFunction):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.state = None

    def open(self, runtime_context: RuntimeContext):
        if False:
            while True:
                i = 10
        state_descriptor = ValueStateDescriptor('state', Types.FLOAT())
        state_ttl_config = StateTtlConfig.new_builder(Time.seconds(1)).set_update_type(StateTtlConfig.UpdateType.OnReadAndWrite).disable_cleanup_in_background().build()
        state_descriptor.enable_time_to_live(state_ttl_config)
        self.state = runtime_context.get_state(state_descriptor)

    def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
        if False:
            while True:
                i = 10
        current = self.state.value()
        if current is None:
            current = 0
        current += value[2]
        self.state.update(current)
        ctx.timer_service().register_event_time_timer(ctx.timestamp() + 2000)

    def on_timer(self, timestamp: int, ctx: 'KeyedProcessFunction.OnTimerContext'):
        if False:
            i = 10
            return i + 15
        yield (ctx.get_current_key(), self.state.value())

class MyTimestampAssigner(TimestampAssigner):

    def extract_timestamp(self, value, record_timestamp: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        return int(value[0])

def event_timer_timer_demo():
    if False:
        for i in range(10):
            print('nop')
    env = StreamExecutionEnvironment.get_execution_environment()
    ds = env.from_collection(collection=[(1000, 'Alice', 110.1), (4000, 'Bob', 30.2), (3000, 'Alice', 20.0), (2000, 'Bob', 53.1), (5000, 'Alice', 13.1), (3000, 'Bob', 3.1), (7000, 'Bob', 16.1), (10000, 'Alice', 20.1)], type_info=Types.TUPLE([Types.LONG(), Types.STRING(), Types.FLOAT()]))
    ds = ds.assign_timestamps_and_watermarks(WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(2)).with_timestamp_assigner(MyTimestampAssigner()))
    ds.key_by(lambda value: value[1]).process(Sum()).print()
    env.execute()
if __name__ == '__main__':
    event_timer_timer_demo()