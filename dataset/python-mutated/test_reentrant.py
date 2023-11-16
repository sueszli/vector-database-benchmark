import orjson

class C:
    c: 'C'

    def __del__(self):
        if False:
            return 10
        orjson.loads('"' + 'a' * 10000 + '"')

def test_reentrant():
    if False:
        print('Hello World!')
    c = C()
    c.c = c
    del c
    orjson.loads('[' + '[],' * 1000 + '[]]')