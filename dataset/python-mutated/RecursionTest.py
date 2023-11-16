count = 0

def recurse():
    if False:
        i = 10
        return i + 15
    global count
    count += 1
    if count < 50:
        recurse()
recurse()