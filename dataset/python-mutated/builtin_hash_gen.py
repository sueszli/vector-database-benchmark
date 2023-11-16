def gen():
    if False:
        i = 10
        return i + 15
    yield
print(type(hash(gen)))
print(type(hash(gen())))