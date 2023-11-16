from nvidia.dali import pipeline_def
import os
import numpy as np
from nvidia.dali import fn
from test_utils import get_dali_extra_path
from nose_utils import raises
import tempfile
test_data_root = get_dali_extra_path()

def _uint8_tensor_to_string(t):
    if False:
        return 10
    return np.array(t).tobytes().decode()

@pipeline_def
def file_properties(files, device):
    if False:
        i = 10
        return i + 15
    (read, _) = fn.readers.file(files=files)
    if device == 'gpu':
        read = read.gpu()
    return fn.get_property(read, key='source_info')

def _test_file_properties(device):
    if False:
        for i in range(10):
            print('nop')
    root_path = os.path.join(test_data_root, 'db', 'single', 'png', '0')
    files = [os.path.join(root_path, i) for i in os.listdir(root_path)]
    p = file_properties(files, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()
    for out in output:
        out = out if device == 'cpu' else out.as_cpu()
        for (source_info, ref) in zip(out, files):
            assert _uint8_tensor_to_string(source_info) == ref

def test_file_properties():
    if False:
        return 10
    for dev in ['cpu', 'gpu']:
        yield (_test_file_properties, dev)

@pipeline_def
def wds_properties(root_path, device, idx_paths):
    if False:
        i = 10
        return i + 15
    read = fn.readers.webdataset(paths=[root_path], index_paths=idx_paths, ext=['jpg'])
    if device == 'gpu':
        read = read.gpu()
    return fn.get_property(read, key='source_info')

def generate_wds_index(root_path, index_path):
    if False:
        for i in range(10):
            print('nop')
    from wds2idx import IndexCreator
    with IndexCreator(root_path, index_path) as ic:
        ic.create_index()

def _test_wds_properties(device, generate_index):
    if False:
        print('Hello World!')
    root_path = os.path.join(get_dali_extra_path(), 'db/webdataset/MNIST/devel-0.tar')
    ref_filenames = ['2000.jpg', '2001.jpg', '2002.jpg', '2003.jpg', '2004.jpg', '2005.jpg', '2006.jpg', '2007.jpg']
    ref_indices = [1536, 4096, 6144, 8704, 11264, 13824, 16384, 18432]
    if generate_index:
        with tempfile.TemporaryDirectory() as idx_dir:
            index_paths = [os.path.join(idx_dir, os.path.basename(root_path) + '.idx')]
            generate_wds_index(root_path, index_paths[0])
            p = wds_properties(root_path, device, index_paths, batch_size=8, num_threads=4, device_id=0)
            p.build()
            output = p.run()
    else:
        p = wds_properties(root_path, device, None, batch_size=8, num_threads=4, device_id=0)
        p.build()
        output = p.run()
    for out in output:
        out = out if device == 'cpu' else out.as_cpu()
        for (source_info, ref_fname, ref_idx) in zip(out, ref_filenames, ref_indices):
            assert _uint8_tensor_to_string(source_info) == f'{root_path}:{ref_idx}:{ref_fname}'

def test_wds_properties():
    if False:
        for i in range(10):
            print('nop')
    for dev in ['cpu', 'gpu']:
        for gen_idx in [True, False]:
            yield (_test_wds_properties, dev, gen_idx)

@pipeline_def
def tfr_properties(root_path, index_path, device):
    if False:
        return 10
    import nvidia.dali.tfrecord as tfrec
    features = {'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ''), 'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1)}
    inputs = fn.readers.tfrecord(path=root_path, index_path=index_path, features=features)
    enc = fn.get_property(inputs['image/encoded'], key='source_info')
    lab = fn.get_property(inputs['image/class/label'], key='source_info')
    if device == 'gpu':
        enc = enc.gpu()
        lab = lab.gpu()
    return (enc, lab)

def _test_tfr_properties(device):
    if False:
        for i in range(10):
            print('nop')
    root_path = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train')
    index_path = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train.idx')
    idx = [0, 171504, 553687, 651500, 820966, 1142396, 1380096, 1532947]
    p = tfr_properties(root_path, index_path, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()
    for out in output:
        out = out if device == 'cpu' else out.as_cpu()
        for (source_info, ref_idx) in zip(out, idx):
            assert _uint8_tensor_to_string(source_info) == f'{root_path} at index {ref_idx}'

def test_tfr_properties():
    if False:
        print('Hello World!')
    for dev in ['cpu', 'gpu']:
        yield (_test_tfr_properties, dev)

@pipeline_def
def es_properties(layouts, device):
    if False:
        while True:
            i = 10
    num_outputs = len(layouts)

    def gen_data():
        if False:
            i = 10
            return i + 15
        yield np.random.rand(num_outputs, 3, 4, 5)
    inp = fn.external_source(source=gen_data, layout=layouts, num_outputs=num_outputs, batch=False, cycle=True, device=device)
    return tuple((fn.get_property(i, key='layout') for i in inp))

def _test_es_properties(device):
    if False:
        i = 10
        return i + 15
    layouts = ['ABC', 'XYZ']
    p = es_properties(layouts, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()
    for (out, lt) in zip(output, layouts):
        out = out if device == 'cpu' else out.as_cpu()
        for sample in out:
            assert _uint8_tensor_to_string(sample), lt

def test_es_properties():
    if False:
        i = 10
        return i + 15
    for dev in ['cpu', 'gpu']:
        yield (_test_es_properties, dev)

@pipeline_def
def improper_property(root_path, device):
    if False:
        i = 10
        return i + 15
    read = fn.readers.webdataset(paths=[root_path], ext=['jpg'])
    return fn.get_property(read, key=["this key doesn't exist"])

@raises(RuntimeError, glob='Unknown property key*')
def _test_improper_property(device):
    if False:
        for i in range(10):
            print('nop')
    root_path = os.path.join(get_dali_extra_path(), 'db/webdataset/MNIST/devel-0.tar')
    p = improper_property(root_path, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    p.run()

def test_improper_property():
    if False:
        i = 10
        return i + 15
    for dev in ['cpu', 'gpu']:
        yield (_test_improper_property, dev)