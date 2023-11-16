@micropython.viper
def get(dest, i: int):
    if False:
        i = 10
        return i + 15
    i += 1
    return dest[i]

@micropython.viper
def set(dest, i: int, val: int):
    if False:
        print('Hello World!')
    i += 1
    dest[i] = val + 1
ar = [i for i in range(3)]
for i in range(len(ar)):
    set(ar, i - 1, i)
print(ar)
for i in range(len(ar)):
    print(get(ar, i - 1))