from cython.cimports.strstr import strstr

def main():
    if False:
        for i in range(10):
            print('nop')
    data: cython.p_char = 'hfvcakdfagbcffvschvxcdfgccbcfhvgcsnfxjh'
    pos = strstr(needle='akd', haystack=data)
    print(pos is not cython.NULL)