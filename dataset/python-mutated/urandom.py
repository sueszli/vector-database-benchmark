import os

def main():
    if False:
        return 10
    print('Random number test')
    r = os.urandom(32)
    print(f'urandom TRNG string is {r}')
main()