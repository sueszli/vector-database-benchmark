from __future__ import annotations
import io
from pre_commit import output

def test_output_write_writes():
    if False:
        for i in range(10):
            print('nop')
    stream = io.BytesIO()
    output.write('hello world', stream)
    assert stream.getvalue() == b'hello world'