import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import os.path
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
from nose_utils import raises
from test_utils import get_dali_extra_path
try:
    tf.compat.v1.disable_eager_execution()
except ModuleNotFoundError:
    pass
test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')
IMG_SIZE = 227
NUM_GPUS = 1

class CommonPipeline(Pipeline):

    def __init__(self, batch_size, num_threads, device_id):
        if False:
            return 10
        super().__init__(batch_size, num_threads, device_id)
        self.decode = ops.decoders.Image(device='mixed', output_type=types.RGB)
        self.resize = ops.Resize(device='gpu', interp_type=types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device='gpu', output_dtype=types.FLOAT, crop=(227, 227), mean=[128.0, 128.0, 128.0], std=[1.0, 1.0, 1.0])
        self.uniform = ops.random.Uniform(range=(0.0, 1.0))
        self.resize_rng = ops.random.Uniform(range=(256, 480))

    def base_define_graph(self, inputs, labels):
        if False:
            i = 10
            return i + 15
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter=self.resize_rng())
        output = self.cmn(images, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        return (output, labels.gpu())

class CaffeReadPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        if False:
            print('Hello World!')
        super().__init__(batch_size, num_threads, device_id)
        self.input = ops.readers.Caffe(path=lmdb_folder, random_shuffle=True, shard_id=device_id, num_shards=num_gpus)

    def define_graph(self):
        if False:
            print('Hello World!')
        (images, labels) = self.input()
        return self.base_define_graph(images, labels)

def get_batch_dali(batch_size, pipe_type, label_type, num_gpus=1):
    if False:
        return 10
    pipes = [pipe_type(batch_size=batch_size, num_threads=2, device_id=device_id, num_gpus=num_gpus) for device_id in range(num_gpus)]
    daliop = dali_tf.DALIIterator()
    images = []
    labels = []
    for d in range(NUM_GPUS):
        with tf.device('/gpu:%i' % d):
            (image, label) = daliop(pipeline=pipes[d], shapes=[(batch_size, 3, 227, 227), ()], dtypes=[tf.int32, label_type], device_id=d)
            images.append(image)
            labels.append(label)
    return [images, labels]

def test_dali_tf_op(pipe_type=CaffeReadPipeline, batch_size=16, iterations=32):
    if False:
        while True:
            i = 10
    test_batch = get_batch_dali(batch_size, pipe_type, tf.int32)
    try:
        from tensorflow.compat.v1 import GPUOptions
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import Session
    except ImportError:
        from tensorflow import GPUOptions
        from tensorflow import ConfigProto
        from tensorflow import Session
    gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = ConfigProto(gpu_options=gpu_options)
    with Session(config=config) as sess:
        for i in range(iterations):
            (imgs, labels) = sess.run(test_batch)
            for label in labels:
                assert np.equal(np.mod(label, 1), 0).all()
                assert (label >= 0).all()
                assert (label <= 999).all()

class PythonOperatorPipeline(Pipeline):

    def __init__(self):
        if False:
            return 10
        super().__init__(1, 1, 0, 0)
        self.python_op = ops.PythonFunction(function=lambda : np.zeros((3, 3, 3)))

    def define_graph(self):
        if False:
            print('Hello World!')
        return self.python_op()

@raises(RuntimeError, glob='Note that some operators * cannot be used with TensorFlow Dataset API and DALIIterator')
def test_python_operator_error():
    if False:
        while True:
            i = 10
    daliop = dali_tf.DALIIterator()
    pipe = PythonOperatorPipeline()
    with tf.device('/cpu:0'):
        _ = daliop(pipeline=pipe, shapes=[(1, 3, 3, 3)], dtypes=[tf.float32], device_id=0)