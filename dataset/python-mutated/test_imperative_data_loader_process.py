import multiprocessing
import queue
import unittest
import numpy as np
from paddle import base
from paddle.base.reader import _reader_process_loop

def get_random_images_and_labels(image_shape, label_shape):
    if False:
        for i in range(10):
            print('nop')
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return (image, label)

def batch_generator_creator(batch_size, batch_num):
    if False:
        for i in range(10):
            print('nop')

    def __reader__():
        if False:
            for i in range(10):
                print('nop')
        for _ in range(batch_num):
            (batch_image, batch_label) = get_random_images_and_labels([batch_size, 784], [batch_size, 1])
            yield (batch_image, batch_label)
    return __reader__

class TestDygraphDataLoaderProcess(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.batch_size = 8
        self.batch_num = 4
        self.epoch_num = 2
        self.capacity = 2

    def test_reader_process_loop(self):
        if False:
            i = 10
            return i + 15

        def __clear_process__(util_queue):
            if False:
                i = 10
                return i + 15
            while True:
                try:
                    util_queue.get_nowait()
                except queue.Empty:
                    break
        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(capacity=self.batch_num + 1, use_multiprocess=True)
            loader.set_batch_generator(batch_generator_creator(self.batch_size, self.batch_num), places=base.CPUPlace())
            loader._data_queue = queue.Queue(self.batch_num + 1)
            _reader_process_loop(loader._batch_reader, loader._data_queue)
            util_queue = multiprocessing.Queue(self.batch_num + 1)
            for _ in range(self.batch_num):
                data = loader._data_queue.get(timeout=10)
                util_queue.put(data)
            clear_process = multiprocessing.Process(target=__clear_process__, args=(util_queue,))
            clear_process.start()

    def test_reader_process_loop_simple_none(self):
        if False:
            return 10

        def none_sample_genarator(batch_num):
            if False:
                for i in range(10):
                    print('nop')

            def __reader__():
                if False:
                    for i in range(10):
                        print('nop')
                for _ in range(batch_num):
                    yield None
            return __reader__
        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(capacity=self.batch_num + 1, use_multiprocess=True)
            loader.set_batch_generator(none_sample_genarator(self.batch_num), places=base.CPUPlace())
            loader._data_queue = queue.Queue(self.batch_num + 1)
            exception = None
            try:
                _reader_process_loop(loader._batch_reader, loader._data_queue)
            except ValueError as ex:
                exception = ex
            self.assertIsNotNone(exception)
if __name__ == '__main__':
    unittest.main()