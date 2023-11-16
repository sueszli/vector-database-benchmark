import sys, itertools

def print_permutations(input):
    if False:
        while True:
            i = 10
    if input:
        permutations = itertools.permutations(input)
        for p in permutations:
            print(''.join(p))
input = 'Hola'
print_permutatins(input)
'\nit should print this words:\nHola\nHoal\nHloa\nHlao\nHaol\nHalo\noHla\noHal\nolHa\nolaH\noaHl\noalH\nlHoa\nlHao\nloHa\nloaH\nlaHo\nlaoH\naHol\naHlo\naoHl\naolH\nalHo\naloH\n'