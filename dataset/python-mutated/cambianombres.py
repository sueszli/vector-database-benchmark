def read_abacus(abacus: []) -> str:
    if False:
        i = 10
        return i + 15
    if check_abacus(abacus) != True:
        return 'Hay algún problema con tu ábaco'
    number = 0
    multiplier = 1000000
    for row in abacus:
        number += multiplier * row.split('---')[0].count('O')
        multiplier = multiplier * 0.1
    return int(number)

def check_abacus(abacus: []) -> bool:
    if False:
        return 10
    if len(abacus) != 7:
        return False
    for row in abacus:
        if row.count('O') != 9 or row.count('---') != 1:
            return False
    return True
'\n * Ejemplo de array y resultado:\n * ["O---OOOOOOOO", <- (1,000,000, 2,000,000... 9,000,000)\n *  "OOO---OOOOOO", <- (100,000, 200,000... 900,000)\n *  "---OOOOOOOOO", <- (10,000, 20,000... 90,000)\n *  "OO---OOOOOOO", <- (1,000, 2,000... 9,000)\n *  "OOOOOOO---OO", <- (100, 200, 300... 900)\n *  "OOOOOOOOO---", <- (10, 20, 30... 90)\n *  "---OOOOOOOOO"] <- (0-9)\n *  \n *  Resultado: 1.302.790\n'
abacus = ['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOOOOOO---OO', 'OOOOOOOOO---', '---OOOOOOOOO']
result = read_abacus(abacus)
print(result)