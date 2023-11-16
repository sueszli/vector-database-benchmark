def _parse(srclines, anon_name):
    if False:
        while True:
            i = 10
    text = ' '.join((l for (_, l) in srclines))
    from ._delim import parse
    yield from parse(text, anon_name)