import unittest
import numpy as np
import paddle
from paddle import base

class TestPyReader(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = 32
        self.epoch_num = 2
        self.sample_num = 10

    def test_returnlist(self):
        if False:
            while True:
                i = 10

        def reader_creator_random_image(height, width):
            if False:
                i = 10
                return i + 15

            def reader():
                if False:
                    for i in range(10):
                        print('nop')
                for i in range(self.sample_num):
                    yield (np.random.uniform(low=0, high=255, size=[height, width]),)
            return reader
        for return_list in [True, False]:
            with base.program_guard(base.Program(), base.Program()):
                image = paddle.static.data(name='image', shape=[-1, 784, 784], dtype='float32')
                reader = base.io.PyReader(feed_list=[image], capacity=4, iterable=True, return_list=return_list)
                user_defined_reader = reader_creator_random_image(784, 784)
                reader.decorate_sample_list_generator(paddle.batch(user_defined_reader, batch_size=self.batch_size), base.core.CPUPlace())
                executor = base.Executor(base.core.CPUPlace())
                executor.run(base.default_main_program())
                for _ in range(self.epoch_num):
                    for data in reader():
                        if return_list:
                            executor.run(feed={'image': data[0][0]})
                        else:
                            executor.run(feed=data)
            with base.dygraph.guard():
                batch_py_reader = base.io.PyReader(capacity=2)
                user_defined_reader = reader_creator_random_image(784, 784)
                batch_py_reader.decorate_sample_generator(user_defined_reader, batch_size=self.batch_size, places=base.core.CPUPlace())
                for epoch in range(self.epoch_num):
                    for (_, data) in enumerate(batch_py_reader()):
                        pass
if __name__ == '__main__':
    unittest.main()