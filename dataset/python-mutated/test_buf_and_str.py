from pyo3_pytests.buf_and_str import BytesExtractor, return_memoryview

def test_extract_bytes():
    if False:
        return 10
    extractor = BytesExtractor()
    message = b'\\(-"-;) A message written in bytes'
    assert extractor.from_bytes(message) == len(message)

def test_extract_str():
    if False:
        while True:
            i = 10
    extractor = BytesExtractor()
    message = '\\(-"-;) A message written as a string'
    assert extractor.from_str(message) == len(message)

def test_extract_str_lossy():
    if False:
        i = 10
        return i + 15
    extractor = BytesExtractor()
    message = '\\(-"-;) A message written with a trailing surrogate \ud800'
    rust_surrogate_len = extractor.from_str_lossy('\ud800')
    assert extractor.from_str_lossy(message) == len(message) - 1 + rust_surrogate_len

def test_extract_buffer():
    if False:
        while True:
            i = 10
    extractor = BytesExtractor()
    message = b'\\(-"-;) A message written in bytes'
    assert extractor.from_buffer(message) == len(message)
    arr = bytearray(b'\\(-"-;) A message written in bytes')
    assert extractor.from_buffer(arr) == len(arr)

def test_return_memoryview():
    if False:
        print('Hello World!')
    view = return_memoryview()
    assert view.readonly
    assert view.contiguous
    assert view.tobytes() == b'hello world'