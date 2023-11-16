def foo(foldnuls, word):
    if False:
        i = 10
        return i + 15
    x = 5 if foldnuls and (not word) else 6
    return x
for (expect, foldnuls, word) in ((6, True, True), (5, True, False), (6, False, True), (6, False, False)):
    assert foo(foldnuls, word) == expect