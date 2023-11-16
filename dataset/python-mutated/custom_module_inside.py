from nvidia.dali.plugin.triton import autoserialize
from nvidia.dali import pipeline_def

@autoserialize
@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def func_under_test():
    if False:
        return 10
    import numpy as np
    return 42