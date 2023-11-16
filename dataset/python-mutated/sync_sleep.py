from time import sleep

def add_one(x):
    if False:
        return 10
    print(f'add_one({x})')
    sleep(5)
    return x + 1

def main(nums=[1, 2, 3]):
    if False:
        while True:
            i = 10
    return list(map(add_one, nums))
if __name__ == '__main__':
    main()