import random
import string
characters = string.ascii_letters + string.digits

def create_id(length: int):
    if False:
        for i in range(10):
            print('nop')
    return ''.join(random.choices(characters, k=length))