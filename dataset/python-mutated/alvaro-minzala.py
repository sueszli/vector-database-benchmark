class AbacusConverter:

    def __init__(self, abacus):
        if False:
            print('Hello World!')
        self.abacus = abacus

    def element_to_number(self, element):
        if False:
            i = 10
            return i + 15
        balls = element.split('---')[0]
        return balls.count('O')

    def convert_to_number(self):
        if False:
            return 10
        number = ''
        for element in self.abacus:
            number += str(self.element_to_number(element))
        if not number:
            number = '0'
        number_with_dots = '{:,}'.format(int(number)).replace(',', '.')
        return number_with_dots
if __name__ == '__main__':
    abacus = ['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOOOOOO---OO', 'OOOOOOOOO---', '---OOOOOOOOO']
    converter = AbacusConverter(abacus)
    resultado = converter.convert_to_number()
    print(resultado)
    abacus = ['----OOOOOOOO', 'OO---OOOOOOO', 'OOO---OOOOOO', 'OOOO---OOOOO', 'OOOOO---OOOO', 'OOOOOO---OOO', 'OOOOOOO---OO']
    converter = AbacusConverter(abacus)
    resultado = converter.convert_to_number()
    print(resultado)