import heapq

def _decorate_source(source):
    if False:
        return 10
    for message in source:
        yield ((message.dt, message.source_id), message)

def date_sorted_sources(*sources):
    if False:
        i = 10
        return i + 15
    '\n    Takes an iterable of sources, generating namestrings and\n    piping their output into date_sort.\n    '
    sorted_stream = heapq.merge(*(_decorate_source(s) for s in sources))
    for (_, message) in sorted_stream:
        yield message