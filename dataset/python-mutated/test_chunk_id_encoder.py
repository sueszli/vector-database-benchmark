from deeplake.constants import ENCODING_DTYPE
from deeplake.util.exceptions import ChunkIdEncoderError
import pytest
from deeplake.core.meta.encode.chunk_id import ChunkIdEncoder

def test_trivial():
    if False:
        print('Hello World!')
    enc = ChunkIdEncoder()
    assert enc.num_chunks == 0
    id1 = enc.generate_chunk_id()
    enc.register_samples(10)
    assert enc.num_chunks == 1
    assert id1 == enc[0]
    assert id1 == enc[9]
    enc.register_samples(10)
    enc.register_samples(9)
    enc.register_samples(1)
    assert enc.num_chunks == 1
    assert enc.num_samples == 30
    assert id1 == enc[10]
    assert id1 == enc[11]
    assert id1 == enc[29]
    id2 = enc.generate_chunk_id()
    enc.register_samples(1)
    id3 = enc.generate_chunk_id()
    enc.register_samples(5)
    assert enc.num_chunks == 3
    assert enc.num_samples == 36
    assert id1 != id2
    assert id2 != id3
    assert id1 == enc[29]
    assert id2 == enc[30]
    assert id3 == enc[31]
    assert enc.translate_index_relative_to_chunks(0) == 0
    assert enc.translate_index_relative_to_chunks(1) == 1
    assert enc.translate_index_relative_to_chunks(29) == 29
    assert enc.translate_index_relative_to_chunks(30) == 0
    assert enc.translate_index_relative_to_chunks(31) == 0
    assert enc.translate_index_relative_to_chunks(35) == 4

def test_failures():
    if False:
        while True:
            i = 10
    enc = ChunkIdEncoder()
    with pytest.raises(ChunkIdEncoderError):
        enc.register_samples(0)
    enc.generate_chunk_id()
    with pytest.raises(ChunkIdEncoderError):
        enc.register_samples(0)
    enc.register_samples(1)
    with pytest.raises(IndexError):
        enc[1]
    enc.generate_chunk_id()
    with pytest.raises(IndexError):
        enc[1]

def test_ids():
    if False:
        for i in range(10):
            print('nop')
    enc = ChunkIdEncoder()
    id = enc.generate_chunk_id()
    assert id.itemsize == ENCODING_DTYPE(1).itemsize
    name = ChunkIdEncoder.name_from_id(id)
    out_id = ChunkIdEncoder.id_from_name(name)
    assert id == out_id