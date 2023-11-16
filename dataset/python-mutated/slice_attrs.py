class A:

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        print(idx.start, idx.stop, idx.step)
try:
    t = A()[1:2]
except:
    print('SKIP')
    raise SystemExit
A()[1:2:3]

class B:

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        try:
            idx.start = 0
        except AttributeError:
            print('AttributeError')
B()[:]