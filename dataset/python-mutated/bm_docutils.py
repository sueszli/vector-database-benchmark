"""
Convert Docutils' documentation from reStructuredText to <format>.
"""
import contextlib
from pathlib import Path
import time
import docutils
from docutils import core
import pyperf
import builtins
try:
    builtins.profile
except AttributeError:

    def profile(func):
        if False:
            print('Hello World!')
        return func
    builtins.profile = profile
try:
    from docutils.utils.math.math2html import Trace
except ImportError:
    pass
else:
    Trace.show = lambda message, channel: ...
DOC_ROOT = (Path(__file__).parent / 'docutils_data' / 'docs').resolve()

@profile
def build_html(doc_root):
    if False:
        while True:
            i = 10
    elapsed = 0
    for file in doc_root.rglob('*.txt'):
        file_contents = file.read_text(encoding='utf-8')
        core.publish_string(source=file_contents, reader_name='standalone', parser_name='restructuredtext', writer_name='html5', settings_overrides={'input_encoding': 'unicode', 'output_encoding': 'unicode', 'report_level': 5})

@profile
def bench_docutils(loops, doc_root):
    if False:
        for i in range(10):
            print('nop')
    for _ in range(loops):
        build_html(doc_root)
if __name__ == '__main__':
    start_p = time.perf_counter()
    bench_docutils(5, DOC_ROOT)
    stop_p = time.perf_counter()
    print('Time elapsed: ', stop_p - start_p)