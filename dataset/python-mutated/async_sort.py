from twisted.internet import defer

@defer.inlineCallbacks
def async_sort(l, key, max_parallel=10):
    if False:
        while True:
            i = 10
    'perform an asynchronous sort with parallel run of the key algorithm\n    '
    sem = defer.DeferredSemaphore(max_parallel)
    try:
        keys = (yield defer.gatherResults([sem.run(key, i) for i in l]))
    except defer.FirstError as e:
        raise e.subFailure.value
    keys = {id(l[i]): v for (i, v) in enumerate(keys)}
    l.sort(key=lambda x: keys[id(x)])