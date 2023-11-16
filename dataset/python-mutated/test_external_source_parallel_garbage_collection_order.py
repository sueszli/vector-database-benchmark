import os
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from test_utils import get_dali_extra_path
import numpy as np
data_root = get_dali_extra_path()
jpeg_file = os.path.join(data_root, 'db', 'single', 'jpeg', '510', 'ship-1083562_640.jpg')
batch_size = 4

def cb(sample_info):
    if False:
        print('Hello World!')
    encoded_img = np.fromfile(jpeg_file, dtype=np.uint8)
    label = 1
    return (encoded_img, np.int32([label]))

@pipeline_def
def simple_pipeline():
    if False:
        while True:
            i = 10
    (jpegs, labels) = fn.external_source(source=cb, num_outputs=2, parallel=True, batch=False)
    images = fn.decoders.image(jpegs, device='cpu')
    return (images, labels)

def _test_no_segfault(method, workers_num):
    if False:
        return 10
    "\n    This may cause segmentation fault on Python teardown if shared memory wrappers managed by the\n    py_pool are garbage collected before pipeline's backend\n    "
    pipe = simple_pipeline(py_start_method=method, py_num_workers=workers_num, batch_size=batch_size, num_threads=4, prefetch_queue_depth=2, device_id=0)
    pipe.build()
    pipe.run()

def test_no_segfault():
    if False:
        i = 10
        return i + 15
    import multiprocessing
    import signal
    for method in ['fork', 'spawn']:
        for _ in range(2):
            for workers_num in range(1, 5):
                mp = multiprocessing.get_context('spawn')
                process = mp.Process(target=_test_no_segfault, args=(method, workers_num))
                process.start()
                process.join()
                if process.exitcode != os.EX_OK:
                    if signal.SIGSEGV == -process.exitcode:
                        raise RuntimeError('Process terminated with signal SIGSEGV')
                    raise RuntimeError('Process exited with {} code'.format(process.exitcode))