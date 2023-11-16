class A:

    def __int__(self):
        if False:
            print('Hello World!')
        return 1 << 100
print(int(A()))