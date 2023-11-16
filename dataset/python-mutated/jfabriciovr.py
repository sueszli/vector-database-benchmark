def ConvertToOct(decimal):
    if False:
        print('Hello World!')
    oct_lst = ConversionProcess(decimal, 8)
    return int(ToString(oct_lst))

def ConvertToHex(decimal):
    if False:
        for i in range(10):
            print('nop')
    hex_lst = []
    num_lst = ConversionProcess(decimal, 16)
    for element in num_lst:
        match element:
            case 10:
                element = 'A'
            case 11:
                element = 'B'
            case 12:
                element = 'C'
            case 13:
                element = 'D'
            case 14:
                element = 'E'
            case 15:
                element = 'F'
        hex_lst.append(element)
    return ToString(hex_lst)

def ConvertToBin(decimal):
    if False:
        print('Hello World!')
    bin_lst = ConversionProcess(decimal, 2)
    return int(ToString(bin_lst))

def ConversionProcess(decimal, base):
    if False:
        while True:
            i = 10
    remainder_lst = []
    while decimal > 0:
        remainder_lst.append(decimal % base)
        decimal //= base
    remainder_lst.reverse()
    return remainder_lst

def ToString(num_lst):
    if False:
        for i in range(10):
            print('nop')
    return ''.join((str(x) for x in num_lst))

def main():
    if False:
        return 10
    decimal = int(input('Enter a decimal number: '))
    print(f'Octal: {ConvertToOct(decimal)}')
    print(f'Hexadecimal: {ConvertToHex(decimal)}')
    print(f'Binary: {ConvertToBin(decimal)}')
main()