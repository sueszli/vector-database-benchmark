from nvidia.dali.plugin.triton import autoserialize
from nvidia.dali import pipeline_def

@autoserialize
@pipeline_def(max_batch_size=1, num_threads=1, device_id=0)
def func_under_test():
    if False:
        return 10
    return 42

@autoserialize
@pipeline_def(max_batch_size=1, num_threads=1, device_id=0)
def func_that_shouldnt_be_here():
    if False:
        return 10
    return 42