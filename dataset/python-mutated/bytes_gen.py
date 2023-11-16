def gen():
    if False:
        i = 10
        return i + 15
    for i in range(4):
        yield i
print(bytes(gen()))