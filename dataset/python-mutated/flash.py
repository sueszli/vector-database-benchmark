from __future__ import print_function
import concurrent.futures
from core.colors import info

def flash(function, links, thread_count):
    if False:
        for i in range(10):
            print('nop')
    'Process the URLs and uses a threadpool to execute a function.'
    links = list(links)
    threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count)
    futures = (threadpool.submit(function, link) for link in links)
    for (i, _) in enumerate(concurrent.futures.as_completed(futures)):
        if i + 1 == len(links) or (i + 1) % thread_count == 0:
            print('%s Progress: %i/%i' % (info, i + 1, len(links)), end='\r')
    print('')