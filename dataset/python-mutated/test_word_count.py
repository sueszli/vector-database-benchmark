from concurrent.futures import ThreadPoolExecutor
import pytest
import word_count

@pytest.fixture(scope='session')
def contents() -> str:
    if False:
        while True:
            i = 10
    text = "\nThe Zen of Python, by Tim Peters\n\nBeautiful is better than ugly.\nExplicit is better than implicit.\nSimple is better than complex.\nComplex is better than complicated.\nFlat is better than nested.\nSparse is better than dense.\nReadability counts.\nSpecial cases aren't special enough to break the rules.\nAlthough practicality beats purity.\nErrors should never pass silently.\nUnless explicitly silenced.\nIn the face of ambiguity, refuse the temptation to guess.\nThere should be one-- and preferably only one --obvious way to do it.\nAlthough that way may not be obvious at first unless you're Dutch.\nNow is better than never.\nAlthough never is often better than *right* now.\nIf the implementation is hard to explain, it's a bad idea.\nIf the implementation is easy to explain, it may be a good idea.\nNamespaces are one honking great idea -- let's do more of those!\n"
    return text * 1000

def test_word_count_rust_parallel(benchmark, contents):
    if False:
        for i in range(10):
            print('nop')
    count = benchmark(word_count.search, contents, 'is')
    assert count == 10000

def test_word_count_rust_sequential(benchmark, contents):
    if False:
        while True:
            i = 10
    count = benchmark(word_count.search_sequential, contents, 'is')
    assert count == 10000

def test_word_count_python_sequential(benchmark, contents):
    if False:
        print('Hello World!')
    count = benchmark(word_count.search_py, contents, 'is')
    assert count == 10000

def run_rust_sequential_twice(executor: ThreadPoolExecutor, contents: str, needle: str) -> int:
    if False:
        return 10
    future_1 = executor.submit(word_count.search_sequential_allow_threads, contents, needle)
    future_2 = executor.submit(word_count.search_sequential_allow_threads, contents, needle)
    result_1 = future_1.result()
    result_2 = future_2.result()
    return result_1 + result_2

def test_word_count_rust_sequential_twice_with_threads(benchmark, contents):
    if False:
        while True:
            i = 10
    executor = ThreadPoolExecutor(max_workers=2)
    count = benchmark(run_rust_sequential_twice, executor, contents, 'is')
    assert count == 20000