def do(mode):
    if False:
        return 10
    if mode == 'rb':
        enc = None
    else:
        enc = 'utf-8'
    f = open('data/utf-8_2.txt', mode=mode, encoding=enc)
    print(f.read(1))
    print(f.read(1))
    print(f.read(2))
    print(f.read(4))
    f.readline()
    print(f.read(1 if mode == 'rt' else 3))
    print(f.read(1 if mode == 'rt' else 4))
    f.close()
do('rb')
do('rt')