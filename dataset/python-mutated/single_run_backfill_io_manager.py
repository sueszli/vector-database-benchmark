from dagster import InputContext, IOManager, OutputContext

class MyIOManager(IOManager):

    def load_input(self, context: InputContext):
        if False:
            while True:
                i = 10
        (start_datetime, end_datetime) = context.asset_partitions_time_window
        return read_data_in_datetime_range(start_datetime, end_datetime)

    def handle_output(self, context: OutputContext, obj):
        if False:
            print('Hello World!')
        (start_datetime, end_datetime) = context.asset_partitions_time_window
        return overwrite_data_in_datetime_range(start_datetime, end_datetime, obj)

def read_data_in_datetime_range(*args):
    if False:
        i = 10
        return i + 15
    ...

def overwrite_data_in_datetime_range(*args):
    if False:
        return 10
    ...