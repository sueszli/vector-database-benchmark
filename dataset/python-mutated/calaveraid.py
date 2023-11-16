from random import sample

def passwordCreator(lenght=8, caps=False, nums=False, symb=False):
    if False:
        for i in range(10):
            print('nop')
    chars = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    if lenght < 8 or lenght > 16:
        print('Wrong lenght (8 -16)')
        return None
    if caps:
        chars.extend([chr(x) for x in range(ord('A'), ord('Z') + 1)])
    if nums:
        chars.extend([chr(x) for x in range(ord('0'), ord('9') + 1)])
    if symb:
        chars.extend(list(',;.:/*-+¡¿?!$&()=@#'))
    return ''.join(sample(chars, lenght))
print(passwordCreator(10, True, True, True))
print(passwordCreator(12, True))
print(passwordCreator(16, True, True))
print(passwordCreator(nums=True, symb=True))