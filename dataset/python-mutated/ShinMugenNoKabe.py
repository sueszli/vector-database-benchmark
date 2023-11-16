ROWS_WEIGHTING = [1000000, 100000, 10000, 1000, 100, 10, 1]

def read_abacus(abacus_sequence: list[str]) -> int:
    if False:
        for i in range(10):
            print('nop')
    if not abacus_sequence or len(abacus_sequence) != 7 or any([not '---' in row or row.count('O') != 9 for row in abacus_sequence]):
        raise ValueError('Introduce un ábaco válido')
    return sum((len(row.split('---')[0]) * ROWS_WEIGHTING[i] for (i, row) in enumerate(abacus_sequence)))
if __name__ == '__main__':
    abacus_sequence_1 = ['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOOOOOO---OO', 'OOOOOOOOO---', '---OOOOOOOOO']
    result1 = read_abacus(abacus_sequence_1)
    assert result1 == 1302790
    print(result1)
    abacus_sequence_2 = ['OOOO---OOOOO', 'O---OOOOOOOO', 'OOOOOOOOO---', 'OOOOOOOO---O', 'OOOO---OOOOO', '---OOOOOOOOO', 'OOOOO---OOOO']
    result2 = read_abacus(abacus_sequence_2)
    assert result2 == 4198405
    print(result2)