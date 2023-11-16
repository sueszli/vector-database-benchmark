from airbyte_cdk.sources.declarative.partition_routers.single_partition_router import SinglePartitionRouter

def test():
    if False:
        i = 10
        return i + 15
    iterator = SinglePartitionRouter(parameters={})
    stream_slices = iterator.stream_slices()
    next_slice = next(stream_slices)
    assert next_slice == dict()