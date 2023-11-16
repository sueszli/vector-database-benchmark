def main():
    if False:
        for i in range(10):
            print('nop')
    for n in range(0, 101):
        print(f'{n} {is_multiple_and_fizz_buzz(n)}')

def is_multiple_and_fizz_buzz(n: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    output = ''
    if n % 3 == 0:
        output += 'Fizz '
    if n % 5 == 0:
        output += 'Buzz'
    return output
if __name__ == '__main__':
    main()