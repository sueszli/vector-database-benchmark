@micropython.viper
def div(x: int, y: int) -> int:
    if False:
        i = 10
        return i + 15
    return x // y

@micropython.viper
def mod(x: int, y: int) -> int:
    if False:
        while True:
            i = 10
    return x % y

def dm(x, y):
    if False:
        i = 10
        return i + 15
    print(div(x, y), mod(x, y))
for x in (-6, 6):
    for y in range(-7, 8):
        if y == 0:
            continue
        dm(x, y)