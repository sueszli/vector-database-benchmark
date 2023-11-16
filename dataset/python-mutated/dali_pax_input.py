import os
from praxis import base_input
from nvidia.dali.plugin import jax as dax
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
training_data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/training/')
validation_data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/testing/')

@pipeline_def(device_id=0, num_threads=4, seed=0)
def mnist_pipeline(data_path, random_shuffle):
    if False:
        for i in range(10):
            print('nop')
    (jpegs, labels) = fn.readers.caffe2(path=data_path, random_shuffle=random_shuffle, name='mnist_caffe2_reader')
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.GRAY)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, std=[255.0], output_layout='HWC')
    labels = labels.gpu()
    labels = fn.reshape(labels, shape=[])
    return (images, labels)

class MnistDaliInput(base_input.BaseInput):

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__post_init__()
        data_path = training_data_path if self.is_training else validation_data_path
        training_pipeline = mnist_pipeline(data_path=data_path, random_shuffle=self.is_training, batch_size=self.batch_size)
        self._iterator = dax.DALIGenericIterator(training_pipeline, output_map=['inputs', 'labels'], reader_name='mnist_caffe2_reader', auto_reset=True)

    def get_next(self):
        if False:
            print('Hello World!')
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator.reset()
            return next(self._iterator)

    def reset(self) -> None:
        if False:
            return 10
        super().reset()
        self._iterator = self._iterator.reset()