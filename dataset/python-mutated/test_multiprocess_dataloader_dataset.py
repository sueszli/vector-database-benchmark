import unittest
import numpy as np
import paddle
from paddle import base
from paddle.io import ChainDataset, ComposeDataset, DataLoader, Dataset, IterableDataset, TensorDataset
IMAGE_SIZE = 32

class RandomDataset(Dataset):

    def __init__(self, sample_num):
        if False:
            i = 10
            return i + 15
        self.sample_num = sample_num

    def __len__(self):
        if False:
            print('Hello World!')
        return self.sample_num

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, 9, (1,)).astype('int64')
        return (image, label)

class RandomIterableDataset(IterableDataset):

    def __init__(self, sample_num):
        if False:
            return 10
        self.sample_num = sample_num

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for i in range(self.sample_num):
            np.random.seed(i)
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, 9, (1,)).astype('int64')
            yield (image, label)

class TestTensorDataset(unittest.TestCase):

    def run_main(self, num_workers, places):
        if False:
            for i in range(10):
                print('nop')
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with base.dygraph.guard(place):
            input_np = np.random.random([16, 3, 4]).astype('float32')
            input = paddle.to_tensor(input_np)
            label_np = np.random.random([16, 1]).astype('int32')
            label = paddle.to_tensor(label_np)
            dataset = TensorDataset([input, label])
            assert len(dataset) == 16
            dataloader = DataLoader(dataset, places=place, num_workers=num_workers, batch_size=1, drop_last=True)
            for (i, (input, label)) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, 3, 4]
                assert label.shape == [1, 1]
                assert isinstance(input, base.core.eager.Tensor)
                assert isinstance(label, base.core.eager.Tensor)
                np.testing.assert_allclose(input.numpy(), input_np[i])
                np.testing.assert_allclose(label.numpy(), label_np[i])

    def test_main(self):
        if False:
            return 10
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)

class TestComposeDataset(unittest.TestCase):

    def test_main(self):
        if False:
            while True:
                i = 10
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        dataset1 = RandomDataset(10)
        dataset2 = RandomDataset(10)
        dataset = ComposeDataset([dataset1, dataset2])
        assert len(dataset) == 10
        for i in range(len(dataset)):
            (input1, label1, input2, label2) = dataset[i]
            (input1_t, label1_t) = dataset1[i]
            (input2_t, label2_t) = dataset2[i]
            np.testing.assert_allclose(input1, input1_t)
            np.testing.assert_allclose(label1, label1_t)
            np.testing.assert_allclose(input2, input2_t)
            np.testing.assert_allclose(label2, label2_t)

class TestRandomSplitApi(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        (dataset1, dataset2) = paddle.io.random_split(range(5), [1, 4])
        self.assertTrue(len(dataset1) == 1)
        self.assertTrue(len(dataset2) == 4)
        elements_list = list(range(5))
        for (_, val) in enumerate(dataset1):
            elements_list.remove(val)
        for (_, val) in enumerate(dataset2):
            elements_list.remove(val)
        self.assertTrue(len(elements_list) == 0)

class TestRandomSplitError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        self.assertRaises(ValueError, paddle.io.random_split, range(5), [3, 8])
        self.assertRaises(ValueError, paddle.io.random_split, range(5), [8])
        self.assertRaises(ValueError, paddle.io.random_split, range(5), [])

class TestSubsetDataset(unittest.TestCase):

    def run_main(self, num_workers, places):
        if False:
            i = 10
            return i + 15
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        input_np = np.random.random([5, 3, 4]).astype('float32')
        input = paddle.to_tensor(input_np)
        label_np = np.random.random([5, 1]).astype('int32')
        label = paddle.to_tensor(label_np)
        dataset = TensorDataset([input, label])
        even_subset = paddle.io.Subset(dataset, [0, 2, 4])
        odd_subset = paddle.io.Subset(dataset, [1, 3])
        assert len(dataset) == 5

        def prepare_dataloader(dataset):
            if False:
                print('Hello World!')
            return DataLoader(dataset, places=places, num_workers=num_workers, batch_size=1, drop_last=True)
        dataloader = prepare_dataloader(dataset)
        dataloader_even = prepare_dataloader(even_subset)
        dataloader_odd = prepare_dataloader(odd_subset)

        def assert_basic(input, label):
            if False:
                while True:
                    i = 10
            assert len(input) == 1
            assert len(label) == 1
            assert input.shape == [1, 3, 4]
            assert label.shape == [1, 1]
            assert isinstance(input, base.core.eager.Tensor)
            assert isinstance(label, base.core.eager.Tensor)
        elements_list = []
        for (_, (input, label)) in enumerate(dataloader()):
            assert_basic(input, label)
            elements_list.append(label)
        for (_, (input, label)) in enumerate(dataloader_even()):
            assert_basic(input, label)
            elements_list.remove(label)
        odd_list = []
        for (_, (input, label)) in enumerate(dataloader_odd()):
            assert_basic(input, label)
            odd_list.append(label)
        self.assertEqual(odd_list, elements_list)

    def test_main(self):
        if False:
            i = 10
            return i + 15
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)

class TestChainDataset(unittest.TestCase):

    def run_main(self, num_workers, places):
        if False:
            print('Hello World!')
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        dataset1 = RandomIterableDataset(10)
        dataset2 = RandomIterableDataset(10)
        dataset = ChainDataset([dataset1, dataset2])
        samples = []
        for data in iter(dataset):
            samples.append(data)
        assert len(samples) == 20
        idx = 0
        for (image, label) in iter(dataset1):
            np.testing.assert_allclose(image, samples[idx][0])
            np.testing.assert_allclose(label, samples[idx][1])
            idx += 1
        for (image, label) in iter(dataset2):
            np.testing.assert_allclose(image, samples[idx][0])
            np.testing.assert_allclose(label, samples[idx][1])
            idx += 1

    def test_main(self):
        if False:
            i = 10
            return i + 15
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)

class NumpyMixTensorDataset(Dataset):

    def __init__(self, sample_num):
        if False:
            while True:
                i = 10
        self.sample_num = sample_num

    def __len__(self):
        if False:
            print('Hello World!')
        return self.sample_num

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, 9, (1,)).astype('int64')
        return (paddle.to_tensor(image, place=paddle.CPUPlace()), label)

class TestNumpyMixTensorDataset(TestTensorDataset):

    def run_main(self, num_workers, places):
        if False:
            while True:
                i = 10
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with base.dygraph.guard(place):
            dataset = NumpyMixTensorDataset(16)
            assert len(dataset) == 16
            dataloader = DataLoader(dataset, places=place, num_workers=num_workers, batch_size=1, drop_last=True)
            for (i, (input, label)) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, IMAGE_SIZE]
                assert label.shape == [1, 1]
                assert isinstance(input, base.core.eager.Tensor)
                assert isinstance(label, base.core.eager.Tensor)

class ComplextDataset(Dataset):

    def __init__(self, sample_num):
        if False:
            i = 10
            return i + 15
        self.sample_num = sample_num

    def __len__(self):
        if False:
            print('Hello World!')
        return self.sample_num

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        return (3.1, 'abc', paddle.to_tensor(np.random.random([IMAGE_SIZE]).astype('float32'), place=paddle.CPUPlace()), [1, np.random.random([2]).astype('float32')], {'a': 2.0, 'b': np.random.random([2]).astype('float32')})

class TestComplextDataset(unittest.TestCase):

    def run_main(self, num_workers):
        if False:
            return 10
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with base.dygraph.guard(place):
            dataset = ComplextDataset(16)
            assert len(dataset) == 16
            dataloader = DataLoader(dataset, places=place, num_workers=num_workers, batch_size=2, drop_last=True)
            for (i, data) in enumerate(dataloader()):
                assert len(data) == 5
                assert data[0].shape == [2]
                assert isinstance(data[1], list)
                assert len(data[1]) == 2
                assert isinstance(data[1][0], str)
                assert isinstance(data[1][1], str)
                assert data[2].shape == [2, IMAGE_SIZE]
                assert isinstance(data[3], list)
                assert data[3][0].shape == [2]
                assert data[3][1].shape == [2, 2]
                assert isinstance(data[4], dict)
                assert data[4]['a'].shape == [2]
                assert data[4]['b'].shape == [2, 2]

    def test_main(self):
        if False:
            print('Hello World!')
        for num_workers in [0, 2]:
            self.run_main(num_workers)

class SingleFieldDataset(Dataset):

    def __init__(self, sample_num):
        if False:
            i = 10
            return i + 15
        self.sample_num = sample_num

    def __len__(self):
        if False:
            print('Hello World!')
        return self.sample_num

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        return np.random.random((2, 3)).astype('float32')

class TestSingleFieldDataset(unittest.TestCase):

    def init_dataset(self):
        if False:
            while True:
                i = 10
        self.sample_num = 16
        self.dataset = SingleFieldDataset(self.sample_num)

    def run_main(self, num_workers):
        if False:
            print('Hello World!')
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with base.dygraph.guard(place):
            self.init_dataset()
            dataloader = DataLoader(self.dataset, places=place, num_workers=num_workers, batch_size=2, drop_last=True)
            for (i, data) in enumerate(dataloader()):
                assert isinstance(data, base.core.eager.Tensor)
                assert data.shape == [2, 2, 3]

    def test_main(self):
        if False:
            i = 10
            return i + 15
        for num_workers in [0, 2]:
            self.run_main(num_workers)

class SingleFieldIterableDataset(IterableDataset):

    def __init__(self, sample_num):
        if False:
            return 10
        self.sample_num = sample_num

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for _ in range(self.sample_num):
            yield np.random.random((2, 3)).astype('float32')

class TestSingleFieldIterableDataset(TestSingleFieldDataset):

    def init_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        self.sample_num = 16
        self.dataset = SingleFieldIterableDataset(self.sample_num)

class TestDataLoaderGenerateStates(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.inputs = [(0, 1), (0, 2), (1, 3)]
        self.outputs = [[1835504127, 1731038949, 1320224556, 2330041505], [2834126987, 2358157858, 1860244682, 1437227251], [457190280, 2660306227, 859341110, 354512857]]

    def test_main(self):
        if False:
            print('Hello World!')
        from paddle.io.dataloader.worker import _generate_states
        for (inp, outp) in zip(self.inputs, self.outputs):
            out = _generate_states(*inp)
            assert out == outp

class TestDatasetWithDropLast(unittest.TestCase):

    def run_main(self, dataset, num_samples, batch_size):
        if False:
            i = 10
            return i + 15
        for num_workers in [0, 1]:
            for drop_last in [True, False]:
                steps = (num_samples + (1 - int(drop_last)) * (batch_size - 1)) // batch_size
                dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, num_workers=num_workers)
                datas = []
                for data in dataloader:
                    datas.append(data)
                assert len(datas) == steps

    def test_map_dataset(self):
        if False:
            print('Hello World!')
        dataset = RandomDataset(10)
        self.run_main(dataset, 10, 3)

    def test_iterable_dataset(self):
        if False:
            return 10
        dataset = RandomIterableDataset(10)
        self.run_main(dataset, 10, 3)
if __name__ == '__main__':
    unittest.main()