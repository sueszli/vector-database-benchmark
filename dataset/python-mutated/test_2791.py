import fitz
import gc
import os
import platform
import sys

def merge_pdf(content: bytes, coverpage: bytes):
    if False:
        while True:
            i = 10
    with fitz.Document(stream=coverpage, filetype='pdf') as coverpage_pdf:
        with fitz.Document(stream=content, filetype='pdf') as content_pdf:
            coverpage_pdf.insert_pdf(content_pdf)
            doc = coverpage_pdf.write()
            return doc

def test_2791():
    if False:
        i = 10
        return i + 15
    '\n    Check for memory leaks.\n    '
    if os.environ.get('PYMUPDF_RUNNING_ON_VALGRIND') == '1':
        print(f'test_2791(): not running because PYMUPDF_RUNNING_ON_VALGRIND=1.')
        return
    stat_type = 'psutil'
    if stat_type == 'tracemalloc':
        import tracemalloc
        tracemalloc.start(10)

        def get_stat():
            if False:
                for i in range(10):
                    print('nop')
            (current, peak) = tracemalloc.get_traced_memory()
            return current
    elif stat_type == 'psutil':
        import psutil
        process = psutil.Process()

        def get_stat():
            if False:
                return 10
            return process.memory_info().rss
    else:

        def get_stat():
            if False:
                print('Hello World!')
            return 0
    n = 1000
    stats = [1] * n
    for i in range(n):
        root = os.path.abspath(f'{__file__}/../../tests/resources')
        with open(f'{root}/test_2791_content.pdf', 'rb') as content_pdf:
            with open(f'{root}/test_2791_coverpage.pdf', 'rb') as coverpage_pdf:
                content = content_pdf.read()
                coverpage = coverpage_pdf.read()
                merge_pdf(content, coverpage)
                sys.stdout.flush()
        gc.collect()
        stats[i] = get_stat()
    print(f'Memory usage stat_type={stat_type!r}.')
    for (i, stat) in enumerate(stats):
        sys.stdout.write(f' {stat}')
    sys.stdout.write('\n')
    first = stats[2]
    last = stats[-1]
    ratio = last / first
    print(f'first={first!r} last={last!r} ratio={ratio!r}')
    if platform.system() != 'Linux':
        print(f'test_2791(): not asserting ratio because not running on Linux.')
    elif not hasattr(fitz, 'mupdf'):
        print(f'test_2791(): not asserting ratio because using classic implementation.')
    elif [int(x) for x in platform.python_version_tuple()[:2]] < [3, 11]:
        print(f'test_2791(): not asserting ratio because python version less than 3.11: platform.python_version()={platform.python_version()!r}.')
    elif stat_type == 'tracemalloc':
        assert ratio > 1 and ratio < 1.6
    elif stat_type == 'psutil':
        assert ratio >= 1 and ratio < 1.015
    else:
        pass