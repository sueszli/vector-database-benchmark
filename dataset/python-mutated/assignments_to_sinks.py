import pyre
from integration_test.taint import source, sink

def indirect(into_global_sink):
    if False:
        return 10
    pyre._global_sink = into_global_sink

def test_indirect():
    if False:
        while True:
            i = 10
    indirect(source())

def test_direct():
    if False:
        for i in range(10):
            print('nop')
    pyre._global_sink = source()