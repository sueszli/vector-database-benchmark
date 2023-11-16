class A:

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        print(idx)
        return idx
s = A()[1:2:3]
print(type(s) is slice)