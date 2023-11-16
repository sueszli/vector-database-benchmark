"""
Topic: 随机数
Desc : 
"""
import random

def random_num():
    if False:
        i = 10
        return i + 15
    values = [1, 2, 3, 4, 5, 6]
    print(random.choice(values))
    print(random.choice(values))
    print(random.choice(values))
    print(random.choice(values))
    print(random.choice(values))
    print(random.sample(values, 2))
    print(random.sample(values, 2))
    print(random.sample(values, 3))
    random.shuffle(values)
    print(values)
    print(random.randint(0, 10))
    print(random.randint(0, 10))
    print(random.randint(0, 10))
    print(random.randint(0, 10))
    print(random.getrandbits(200))
    random.seed()
    random.seed(12345)
    random.seed(b'bytedata')
if __name__ == '__main__':
    random_num()