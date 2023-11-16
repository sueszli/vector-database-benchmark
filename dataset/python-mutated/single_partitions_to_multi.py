from dagster import DynamicPartitionsDefinition, MultiPartitionsDefinition, StaticPartitionsDefinition, asset
abc_def = StaticPartitionsDefinition(['a', 'b', 'c'])
composite = MultiPartitionsDefinition({'abc': abc_def, '123': StaticPartitionsDefinition(['1', '2', '3'])})
composite2 = MultiPartitionsDefinition({'abc': abc_def, '123': DynamicPartitionsDefinition(name='testing123')})

@asset(partitions_def=abc_def)
def single_partitions(context):
    if False:
        i = 10
        return i + 15
    return 1

@asset(partitions_def=composite)
def multi_partitions(single_partitions):
    if False:
        print('Hello World!')
    return 1

@asset(partitions_def=composite2)
def multi_partitions_dynamic(single_partitions):
    if False:
        while True:
            i = 10
    return 1