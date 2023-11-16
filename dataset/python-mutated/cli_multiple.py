import hug

@hug.cli()
def add(numbers: list=None):
    if False:
        for i in range(10):
            print('nop')
    return sum([int(number) for number in numbers])
if __name__ == '__main__':
    add.interface.cli()