import tempfile
import os.path
import pytest
from unittest import TestCase
import shutil
from bigdl.dllib.nncontext import *
from bigdl.orca.data.image import write_tfrecord, read_tfrecord

class TestTFRecord(TestCase):

    def setup_method(self, method):
        if False:
            return 10
        self.resource_path = os.path.join(os.path.split(__file__)[0], '../resources')

    def test_write_read_imagenet(self):
        if False:
            print('Hello World!')
        raw_data = os.path.join(self.resource_path, 'imagenet_to_tfrecord')
        temp_dir = tempfile.mkdtemp()
        try:
            write_tfrecord(format='imagenet', imagenet_path=raw_data, output_path=temp_dir)
            data_dir = os.path.join(temp_dir, 'train')
            train_dataset = read_tfrecord(format='imagenet', path=data_dir, is_training=True)
            train_dataset.take(1)
        finally:
            shutil.rmtree(temp_dir)
if __name__ == '__main__':
    pytest.main([__file__])