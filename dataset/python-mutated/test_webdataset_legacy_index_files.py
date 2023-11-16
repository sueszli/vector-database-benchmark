from nvidia.dali import pipeline_def, fn
import os
import glob
from test_utils import get_dali_extra_path
test_data_root = os.path.join(get_dali_extra_path(), 'db', 'webdataset', 'legacy_index_formats')

@pipeline_def(batch_size=8, num_threads=4, device_id=0)
def wds_index_file_pipeline(idx_path, device):
    if False:
        i = 10
        return i + 15
    (jpg, cls) = fn.readers.webdataset(paths=[os.path.join(test_data_root, 'data.tar')], index_paths=[idx_path], ext=['jpg', 'cls'])
    if device == 'gpu':
        jpg = jpg.gpu()
        cls = cls.gpu()
    return (jpg, cls)

def _test_wds_index_file_pipeline(idx_path, device):
    if False:
        print('Hello World!')
    p = wds_index_file_pipeline(idx_path, device)
    p.build()
    p.run()

def test_wds_index_file_pipeline():
    if False:
        while True:
            i = 10
    idx_files = glob.glob(test_data_root + '/*.idx')
    for idx_path in idx_files:
        for device in ['cpu', 'gpu']:
            yield (_test_wds_index_file_pipeline, idx_path, device)