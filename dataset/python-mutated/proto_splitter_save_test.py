"""Tests for SavedModel save using experimental_image_format."""
import os
from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.module import module
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.tools.proto_splitter import constants

class ProtoSplitterSaveTest(test.TestCase, parameterized.TestCase):

    def test_save_experimental_image_format(self):
        if False:
            for i in range(10):
                print('nop')
        root = module.Module()
        root.c = constant_op.constant(np.random.random_sample([150, 150]))
        root.get_c = def_function.function(lambda : root.c)
        save_dir = os.path.join(self.get_temp_dir(), 'chunked_model')
        constants.debug_set_max_size(80000)
        options = save_options.SaveOptions(experimental_image_format=True)
        save.save(root, save_dir, signatures=root.get_c.get_concrete_function(), options=options)
        self.assertTrue(os.path.exists(save_dir + '/saved_model.cpb'))

    def test_save_experimental_image_format_not_chunked(self):
        if False:
            while True:
                i = 10
        root = module.Module()
        root.c = constant_op.constant(np.random.random_sample([150, 150]))
        root.get_c = def_function.function(lambda : root.c)
        save_dir = os.path.join(self.get_temp_dir(), 'not_chunked_model')
        constants.debug_set_max_size(1 << 31)
        options = save_options.SaveOptions(experimental_image_format=True)
        save.save(root, save_dir, signatures=root.get_c.get_concrete_function(), options=options)
        self.assertTrue(os.path.exists(save_dir + '/saved_model.pb'))
if __name__ == '__main__':
    test.main()