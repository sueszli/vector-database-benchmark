import sys
import unittest
import paddle
from paddle.io import Dataset
from paddle.vision import transforms

class TestDatasetAbstract(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        dataset = Dataset()
        try:
            d = dataset[0]
            self.assertTrue(False)
        except NotImplementedError:
            pass
        try:
            l = len(dataset)
            self.assertTrue(False)
        except NotImplementedError:
            pass

class TestDatasetWithDiffOutputPlace(unittest.TestCase):

    def get_dataloader(self, num_workers):
        if False:
            print('Hello World!')
        dataset = paddle.vision.datasets.MNIST(mode='test', transform=transforms.Compose([transforms.CenterCrop(20), transforms.RandomResizedCrop(14), transforms.Normalize(), transforms.ToTensor()]))
        loader = paddle.io.DataLoader(dataset, batch_size=32, num_workers=num_workers, shuffle=True)
        return loader

    def run_check_on_cpu(self):
        if False:
            while True:
                i = 10
        paddle.set_device('cpu')
        loader = self.get_dataloader(1)
        for (image, label) in loader:
            self.assertTrue(image.place.is_cpu_place())
            self.assertTrue(label.place.is_cpu_place())
            break

    def test_single_process(self):
        if False:
            print('Hello World!')
        self.run_check_on_cpu()
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            loader = self.get_dataloader(0)
            for (image, label) in loader:
                self.assertTrue(image.place.is_gpu_place())
                self.assertTrue(label.place.is_cuda_pinned_place())
                break

    def test_multi_process(self):
        if False:
            print('Hello World!')
        if sys.platform != 'darwin' and sys.platform != 'win32':
            self.run_check_on_cpu()
            if paddle.is_compiled_with_cuda():
                paddle.set_device('gpu')
                loader = self.get_dataloader(1)
                for (image, label) in loader:
                    self.assertTrue(image.place.is_cuda_pinned_place())
                    self.assertTrue(label.place.is_cuda_pinned_place())
                    break
if __name__ == '__main__':
    unittest.main()