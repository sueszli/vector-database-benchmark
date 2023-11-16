import gc
from multiprocessing import Pool

def pool_worker(item):
    if False:
        while True:
            i = 10
    return {'a': 1}

def pool_indexer(path):
    if False:
        while True:
            i = 10
    item_count = 0
    with Pool(processes=4) as pool:
        for _ in pool.imap(pool_worker, range(1, 200), chunksize=10):
            item_count = item_count + 1
gc.disable()
pool_indexer(10)
gc.enable()