import time
current_time = int(time.time())
seed = current_time % 100

def generate_random_number():
    if False:
        for i in range(10):
            print('nop')
    global seed
    a = 987654321
    c = 123456789
    m = 2 ** 16
    seed = (a * seed + c) % m
    return seed % 101
random_number = generate_random_number()
print(random_number)