import logging
from multiprocessing.pool import ThreadPool
try:
    from lz4.block import compress as lz4_compress, decompress as lz4_decompress
    lz4_compressHC = lambda _str: lz4_compress(_str, mode='high_compression')
except ImportError as e:
    from lz4 import compress as lz4_compress, compressHC as lz4_compressHC, decompress as lz4_decompress
from ._config import ENABLE_PARALLEL, LZ4_HIGH_COMPRESSION, LZ4_WORKERS, LZ4_N_PARALLEL, LZ4_MINSZ_PARALLEL, BENCHMARK_MODE
logger = logging.getLogger(__name__)
_compress_thread_pool = None

def enable_parallel_lz4(mode):
    if False:
        i = 10
        return i + 15
    '\n    Set the global multithread compression mode\n\n    Parameters\n    ----------\n        mode: `bool`\n            True: Use parallel compression. False: Use sequential compression\n    '
    global ENABLE_PARALLEL
    ENABLE_PARALLEL = bool(mode)
    logger.info('Setting parallelisation mode to {}'.format('multi-threaded' if mode else 'single-threaded'))

def set_compression_pool_size(pool_size):
    if False:
        while True:
            i = 10
    '\n    Set the size of the compression workers thread pool.\n    If the pool is already created, it waits until all jobs are finished, and then proceeds with setting the new size.\n\n    Parameters\n    ----------\n        pool_size : `int`\n            The size of the pool (must be a positive integer)\n\n    Returns\n    -------\n    `None`\n    '
    pool_size = int(pool_size)
    if pool_size < 1:
        raise ValueError('The compression thread pool size cannot be of size {}'.format(pool_size))
    global _compress_thread_pool
    if _compress_thread_pool is not None:
        _compress_thread_pool.close()
        _compress_thread_pool.join()
    _compress_thread_pool = ThreadPool(pool_size)

def compress_array(str_list, withHC=LZ4_HIGH_COMPRESSION):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compress an array of strings\n\n    Parameters\n    ----------\n        str_list: `list[str]`\n            The input list of strings which need to be compressed.\n        withHC: `bool`\n            This flag controls whether lz4HC will be used.\n\n    Returns\n    -------\n    `list[str`\n    The list of the compressed strings.\n    '
    global _compress_thread_pool
    if not str_list:
        return str_list
    do_compress = lz4_compressHC if withHC else lz4_compress

    def can_parallelize_strlist(strlist):
        if False:
            while True:
                i = 10
        return len(strlist) > LZ4_N_PARALLEL and len(strlist[0]) > LZ4_MINSZ_PARALLEL
    use_parallel = ENABLE_PARALLEL and withHC or can_parallelize_strlist(str_list)
    if BENCHMARK_MODE or use_parallel:
        if _compress_thread_pool is None:
            _compress_thread_pool = ThreadPool(LZ4_WORKERS)
        return _compress_thread_pool.map(do_compress, str_list)
    return [do_compress(s) for s in str_list]

def compress(_str):
    if False:
        return 10
    '\n    Compress a string\n\n    By default LZ4 mode is standard in interactive mode,\n    and high compresion in applications/scripts\n    '
    return lz4_compress(_str)

def compressHC(_str):
    if False:
        i = 10
        return i + 15
    '\n    HC compression\n    '
    return lz4_compressHC(_str)

def compressHC_array(str_list):
    if False:
        return 10
    '\n    HC compression\n    '
    return compress_array(str_list, withHC=True)

def decompress(_str):
    if False:
        return 10
    '\n    Decompress a string\n    '
    return lz4_decompress(_str)

def decompress_array(str_list):
    if False:
        i = 10
        return i + 15
    '\n    Decompress a list of strings\n    '
    global _compress_thread_pool
    if not str_list:
        return str_list
    if not ENABLE_PARALLEL or len(str_list) <= LZ4_N_PARALLEL:
        return [lz4_decompress(chunk) for chunk in str_list]
    if _compress_thread_pool is None:
        _compress_thread_pool = ThreadPool(LZ4_WORKERS)
    return _compress_thread_pool.map(lz4_decompress, str_list)