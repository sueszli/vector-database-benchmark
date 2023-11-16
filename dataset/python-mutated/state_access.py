from pyflink.common import Time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig

class Sum(KeyedProcessFunction):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = None

    def open(self, runtime_context: RuntimeContext):
        if False:
            return 10
        state_descriptor = ValueStateDescriptor('state', Types.FLOAT())
        state_ttl_config = StateTtlConfig.new_builder(Time.seconds(1)).set_update_type(StateTtlConfig.UpdateType.OnReadAndWrite).disable_cleanup_in_background().build()
        state_descriptor.enable_time_to_live(state_ttl_config)
        self.state = runtime_context.get_state(state_descriptor)

    def process_element(self, value, ctx: 'KeyedProcessFunction.Context'):
        if False:
            return 10
        current = self.state.value()
        if current is None:
            current = 0
        current += value[1]
        self.state.update(current)
        yield (value[0], current)

def state_access_demo():
    if False:
        for i in range(10):
            print('nop')
    env = StreamExecutionEnvironment.get_execution_environment()
    ds = env.from_collection(collection=[('Alice', 110.1), ('Bob', 30.2), ('Alice', 20.0), ('Bob', 53.1), ('Alice', 13.1), ('Bob', 3.1), ('Bob', 16.1), ('Alice', 20.1)], type_info=Types.TUPLE([Types.STRING(), Types.FLOAT()]))
    ds.key_by(lambda value: value[0]).process(Sum()).print()
    env.execute()
if __name__ == '__main__':
    state_access_demo()