import time

def rand(seed):
    if False:
        while True:
            i = 10
    random_seed = 7 ** 5 * seed % (-1 + 2 ** 31)
    print(random_seed % 101)
seed = int(time.time())
rand(seed)