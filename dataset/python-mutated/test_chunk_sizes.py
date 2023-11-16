import numpy as np
from deeplake.constants import KB

def _assert_num_chunks(tensor, expected_num_chunks):
    if False:
        return 10
    chunk_engine = tensor.chunk_engine
    actual_num_chunks = chunk_engine.chunk_id_encoder.num_chunks
    assert actual_num_chunks == expected_num_chunks

def _create_tensors(ds):
    if False:
        i = 10
        return i + 15
    images = ds.create_tensor('images', htype='image', sample_compression=None, max_chunk_size=32 * KB, tiling_threshold=16 * KB)
    labels = ds.create_tensor('labels', htype='class_label', max_chunk_size=32 * KB, tiling_threshold=16 * KB)
    return (images, labels)

def _append_tensors(images, labels):
    if False:
        print('Hello World!')
    for i in range(100):
        x = np.ones((28, 28), dtype=np.uint8) * i
        y = np.uint32(i)
        images.append(x)
        labels.append(y)

def _extend_tensors(images, labels):
    if False:
        while True:
            i = 10
    images.extend(np.ones((100, 28, 28), dtype=np.uint8))
    labels.extend(np.ones(100, dtype=np.uint32))

def _clear_tensors(images, labels):
    if False:
        for i in range(10):
            print('nop')
    images.clear()
    labels.clear()

def test_append(memory_ds):
    if False:
        return 10
    ds = memory_ds
    (images, labels) = _create_tensors(ds)
    _append_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 5)
    _append_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 10)
    _append_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 15)
    assert len(ds) == 300

def test_extend(memory_ds):
    if False:
        i = 10
        return i + 15
    ds = memory_ds
    (images, labels) = _create_tensors(ds)
    _extend_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 5)
    _extend_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 10)
    _extend_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 15)
    assert len(ds) == 300

def test_extend_and_append(memory_ds):
    if False:
        print('Hello World!')
    ds = memory_ds
    (images, labels) = _create_tensors(ds)
    _extend_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 5)
    _append_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 10)
    _extend_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 15)
    _append_tensors(images, labels)
    _assert_num_chunks(labels, 1)
    _assert_num_chunks(images, 20)
    assert len(ds) == 400

def test_clear(memory_ds):
    if False:
        return 10
    ds = memory_ds
    (images, labels) = _create_tensors(ds)
    _clear_tensors(images, labels)
    _assert_num_chunks(labels, 0)
    _assert_num_chunks(images, 0)