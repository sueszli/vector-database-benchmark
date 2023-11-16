import random
import string

def generate_salt():
    if False:
        print('Hello World!')
    return ''.join(random.choices(string.hexdigits, k=36))