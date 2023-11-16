import os
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIImageType
import nvidia.dali.fn as fn
import nvidia.dali.plugin_manager as plugin_manager
plugin_manager.load_library('./build/libnaivehistogram.so')
dali_extra_path = os.environ['DALI_EXTRA_PATH']
test_file_list = [dali_extra_path + '/db/single/jpeg/100/swan-3584559_640.jpg', dali_extra_path + '/db/single/jpeg/113/snail-4368154_1280.jpg', dali_extra_path + '/db/single/jpeg/100/swan-3584559_640.jpg', dali_extra_path + '/db/single/jpeg/113/snail-4368154_1280.jpg', dali_extra_path + '/db/single/jpeg/100/swan-3584559_640.jpg', dali_extra_path + '/db/single/jpeg/113/snail-4368154_1280.jpg']

@pipeline_def
def naive_hist_pipe():
    if False:
        for i in range(10):
            print('nop')
    (img, _) = fn.readers.file(files=test_file_list)
    img = fn.decoders.image(img, device='mixed', output_type=DALIImageType.GRAY)
    img = img.gpu()
    img = fn.naive_histogram(img, n_bins=24)
    return img
pipe = naive_hist_pipe(batch_size=2, num_threads=1, device_id=0)
pipe.build()
out = pipe.run()
print(out[0].as_cpu().as_array())