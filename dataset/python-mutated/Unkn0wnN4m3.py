def m_table(number: int) -> None:
    if False:
        while True:
            i = 10
    txt = '{} x {} = {}'
    for i in range(1, 11):
        result = i * number
        print(txt.format(number, i, result))

def main():
    if False:
        print('Hello World!')
    try:
        n = int(input('Give me a number: '))
    except ValueError as e:
        print('Invalid number:', e)
    else:
        m_table(n)
if __name__ == '__main__':
    main()