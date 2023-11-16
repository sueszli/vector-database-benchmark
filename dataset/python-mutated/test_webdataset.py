import glob
import io
import os
import tarfile
import pytest
import ray
from ray.tests.conftest import *

class TarWriter:

    def __init__(self, path):
        if False:
            return 10
        self.path = path
        self.tar = tarfile.open(path, 'w')

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        self.tar.close()

    def write(self, name, data):
        if False:
            return 10
        f = self.tar.tarinfo()
        f.name = name
        f.size = len(data)
        self.tar.addfile(f, io.BytesIO(data))

def test_webdataset_read(ray_start_2_cpus, tmp_path):
    if False:
        while True:
            i = 10
    path = os.path.join(tmp_path, 'bar_000000.tar')
    with TarWriter(path) as tf:
        for i in range(100):
            tf.write(f'{i}.a', str(i).encode('utf-8'))
            tf.write(f'{i}.b', str(i ** 2).encode('utf-8'))
    assert os.path.exists(path)
    assert len(glob.glob(f'{tmp_path}/*.tar')) == 1
    ds = ray.data.read_webdataset(paths=[str(tmp_path)], parallelism=1)
    samples = ds.take(100)
    assert len(samples) == 100
    for (i, sample) in enumerate(samples):
        assert isinstance(sample, dict), sample
        assert sample['__key__'] == str(i)
        assert sample['a'].decode('utf-8') == str(i)
        assert sample['b'].decode('utf-8') == str(i ** 2)

def test_webdataset_suffixes(ray_start_2_cpus, tmp_path):
    if False:
        return 10
    path = os.path.join(tmp_path, 'bar_000000.tar')
    with TarWriter(path) as tf:
        for i in range(100):
            tf.write(f'{i}.txt', str(i).encode('utf-8'))
            tf.write(f'{i}.test.txt', str(i ** 2).encode('utf-8'))
            tf.write(f'{i}.cls', str(i ** 2).encode('utf-8'))
            tf.write(f'{i}.test.cls2', str(i ** 2).encode('utf-8'))
    assert os.path.exists(path)
    assert len(glob.glob(f'{tmp_path}/*.tar')) == 1
    ds = ray.data.read_webdataset(paths=[str(tmp_path)], parallelism=1, suffixes=['txt', 'cls'])
    samples = ds.take(100)
    assert len(samples) == 100
    for (i, sample) in enumerate(samples):
        assert set(sample.keys()) == {'__url__', '__key__', 'txt', 'cls'}
    ds = ray.data.read_webdataset(paths=[str(tmp_path)], parallelism=1, suffixes=['*.txt', '*.cls'])
    samples = ds.take(100)
    assert len(samples) == 100
    for (i, sample) in enumerate(samples):
        assert set(sample.keys()) == {'__url__', '__key__', 'txt', 'cls', 'test.txt'}

    def select(name):
        if False:
            for i in range(10):
                print('nop')
        return name.endswith('txt')
    ds = ray.data.read_webdataset(paths=[str(tmp_path)], parallelism=1, suffixes=select)
    samples = ds.take(100)
    assert len(samples) == 100
    for (i, sample) in enumerate(samples):
        assert set(sample.keys()) == {'__url__', '__key__', 'txt', 'test.txt'}

    def renamer(name):
        if False:
            while True:
                i = 10
        result = name.replace('txt', 'text')
        print('***', name, result)
        return result
    ds = ray.data.read_webdataset(paths=[str(tmp_path)], parallelism=1, filerename=renamer)
    samples = ds.take(100)
    assert len(samples) == 100
    for (i, sample) in enumerate(samples):
        assert set(sample.keys()) == {'__url__', '__key__', 'text', 'cls', 'test.text', 'test.cls2'}

def test_webdataset_write(ray_start_2_cpus, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    print(ray.available_resources())
    data = [dict(__key__=str(i), a=str(i), b=str(i ** 2)) for i in range(100)]
    ds = ray.data.from_items(data).repartition(1)
    ds.write_webdataset(path=tmp_path, try_create_dir=True)
    paths = glob.glob(f'{tmp_path}/*.tar')
    assert len(paths) == 1
    with open(paths[0], 'rb') as stream:
        tf = tarfile.open(fileobj=stream)
        for i in range(100):
            assert tf.extractfile(f'{i}.a').read().decode('utf-8') == str(i)
            assert tf.extractfile(f'{i}.b').read().decode('utf-8') == str(i ** 2)

def custom_decoder(sample):
    if False:
        i = 10
        return i + 15
    for (key, value) in sample.items():
        if key == 'png':
            assert not isinstance(value, bytes)
        elif key.endswith('custom'):
            sample[key] = 'custom-value'
    return sample

def test_webdataset_coding(ray_start_2_cpus, tmp_path):
    if False:
        while True:
            i = 10
    import numpy as np
    import PIL.Image
    import torch
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    dstruct = dict(a=[1], b=dict(c=2), d='hello')
    ttensor = torch.tensor([1, 2, 3]).numpy()
    sample = {'__key__': 'foo', 'jpg': image, 'gray.png': gray, 'mp': dstruct, 'json': dstruct, 'pt': ttensor, 'und': b'undecoded', 'custom': b'nothing'}
    data = [sample]
    ds = ray.data.from_items(data).repartition(1)
    ds.write_webdataset(path=tmp_path, try_create_dir=True)
    paths = glob.glob(f'{tmp_path}/*.tar')
    assert len(paths) == 1
    path = paths[0]
    assert os.path.exists(path)
    ds = ray.data.read_webdataset(paths=[str(tmp_path)], parallelism=1)
    samples = ds.take(1)
    assert len(samples) == 1
    for sample in samples:
        assert isinstance(sample, dict), sample
        assert sample['__key__'] == 'foo'
        assert isinstance(sample['jpg'], np.ndarray)
        assert sample['jpg'].shape == (100, 100, 3)
        assert isinstance(sample['gray.png'], np.ndarray)
        assert sample['gray.png'].shape == (100, 100)
        assert isinstance(sample['mp'], dict)
        assert sample['mp']['a'] == [1]
        assert sample['mp']['b']['c'] == 2
        assert isinstance(sample['json'], dict)
        assert sample['json']['a'] == [1]
        assert isinstance(sample['pt'], np.ndarray)
        assert sample['pt'].tolist() == [1, 2, 3]
    ds = ray.data.read_webdataset(paths=[str(tmp_path)], parallelism=1, decoder=['PIL', custom_decoder])
    samples = ds.take(1)
    assert len(samples) == 1
    for sample in samples:
        assert isinstance(sample, dict), sample
        assert sample['__key__'] == 'foo'
        assert isinstance(sample['jpg'], PIL.Image.Image)
        assert isinstance(sample['gray.png'], PIL.Image.Image)
        assert isinstance(sample['und'], bytes)
        assert sample['und'] == b'undecoded'
        assert sample['custom'] == 'custom-value'
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))