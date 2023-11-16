try:
    import esp32
except ImportError:
    print('SKIP')
    raise SystemExit

def chk_heap(kind, regions):
    if False:
        while True:
            i = 10
    chk = [True, True, True, True]
    for r in regions:
        chk = [chk[0] and r[0] > 0, chk[1] and r[1] <= r[0], chk[2] and r[2] <= r[0], chk[3] and r[3] <= r[0]]
    print(kind, chk)
regions = esp32.idf_heap_info(esp32.HEAP_DATA)
print('HEAP_DATA >2:', len(regions) > 2)
chk_heap('HEAP_DATA', regions)
regions = esp32.idf_heap_info(esp32.HEAP_EXEC)
print('HEAP_EXEC >2:', len(regions) > 2)
chk_heap('HEAP_EXEC', regions)
print(esp32.idf_heap_info(-1))