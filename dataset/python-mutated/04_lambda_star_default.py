def pack(width, data):
    if False:
        return 10
    return (width, data)
packs = {w: lambda *data, width=w: pack(width, data) for w in (1, 2, 4)}
assert packs[1]('a') == (1, ('a',))
assert packs[2]('b') == (2, ('b',))
assert packs[4]('c') == (4, ('c',))