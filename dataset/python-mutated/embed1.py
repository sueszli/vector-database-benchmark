"""This tests standard embedding, automatically detecting the module and
local namespaces."""
f = set([1, 2, 3, 4, 5])

def bar(foo):
    if False:
        while True:
            i = 10
    import IPython
    IPython.embed(banner1='check f in globals, foo in locals')
bar(f)