__license__ = 'GPL v3'
__copyright__ = '2010, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
'\nMeasure memory usage of the current process.\n\nThe key function is memory() which returns the current memory usage in MB.\nYou can pass a number to memory and it will be subtracted from the returned\nvalue.\n'
import gc, os

def get_memory():
    if False:
        for i in range(10):
            print('nop')
    'Return memory usage in bytes'
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss

def memory(since=0.0):
    if False:
        for i in range(10):
            print('nop')
    'Return memory used in MB. The value of since is subtracted from the used memory'
    ans = get_memory()
    ans /= float(1024 ** 2)
    return ans - since

def gc_histogram():
    if False:
        print('Hello World!')
    'Returns per-class counts of existing objects.'
    result = {}
    for o in gc.get_objects():
        t = type(o)
        count = result.get(t, 0)
        result[t] = count + 1
    return result

def diff_hists(h1, h2):
    if False:
        for i in range(10):
            print('nop')
    'Prints differences between two results of gc_histogram().'
    for k in h1:
        if k not in h2:
            h2[k] = 0
        if h1[k] != h2[k]:
            print('%s: %d -> %d (%s%d)' % (k, h1[k], h2[k], h2[k] > h1[k] and '+' or '', h2[k] - h1[k]))