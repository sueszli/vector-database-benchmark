"""
Solution author: https://github.com/0xTrivi
Challenge number 20 of @Mouredev
Source repository: https://github.com/mouredev/retos-programacion-2023

Statement:

The new "The Legend of Zelda: Tears of the Kingdom" is now available!
Create a program that draws a Triforce from "Zelda" using asterisks.

You should provide the number of rows for the triangles with a positive integer (n).
Each triangle will calculate its largest row using the formula 2n-1.
Example: Triforce 2

"""

def print_spaces(number: int):
    if False:
        while True:
            i = 10
    'Prints in a line of text the number of spaces set.'
    for j in range(number):
        print('', end=' ')

def print_asterisks(number: int):
    if False:
        while True:
            i = 10
    'Prints on a line of text the number of asterisks set \n    and colors them yellow.'
    for k in range(number):
        print('\x1b[33m*\x1b[0m', end=' ')

def print_Power(size: int, lastRowSize: int):
    if False:
        print('Hello World!')
    'Print the upper Triforce triangle.'
    for i in range(size):
        print_spaces(lastRowSize - i)
        print_asterisks(i + 1)
        print_spaces(lastRowSize - i)
        print(' ')

def print_Wisdom_and_Courage(numberOfRows: int, lastRowSize: int):
    if False:
        return 10
    'Print the two lower triangles of the Triforce.'
    for i in range(numberOfRows):
        print_spaces(numberOfRows - i - 1)
        print_asterisks(i + 1)
        print_spaces(lastRowSize - 2 * i - 1)
        print_asterisks(i + 1)
        print_spaces(numberOfRows - i - 1)
        print(' ')

def print_Triforce(numberOfRows: int, lastRowSize: int):
    if False:
        for i in range(10):
            print('nop')
    'Prints a yellow Triforce with the number of rows \n    established for each triangle.'
    print_Power(numberOfRows, lastRowSize)
    print_Wisdom_and_Courage(numberOfRows, lastRowSize)
    print(' ')

def what_size() -> int:
    if False:
        while True:
            i = 10
    'Allows the user to set the number of rows for each \n    triangle of the Triforce.'
    size = False
    print('How big do you want your triforce?:')
    while size == False:
        number = int(input(' '))
        print(' ')
        if number > 0 and type(number) == int:
            size = True
            print('')
        else:
            print('Please! write a positive integer number:')
    print('')
    return number

def print_title():
    if False:
        while True:
            i = 10
    'Print the tittle to the program.'
    print('\n                  T~~\n                  |\n                /"|\n        T~~     |\'|  T~~\n    T~~ |     T~ WWWW|\n    |  /"\\    |  |  |/\\T~~\n    /"\\ WWW  /"\\ |\' |WW|  T~~\n    WWWWW/\\| /   \\|\'/\\|/"\'|\n    |   /__\\/]WWW[\\/__\\WWWW\n    |"  WWWW\'|I_I|\'WWWW\'  |\n    |   |\' |/  -  \\|\' |\'  |\n    |\'  |  |LI=H=LI|\' |   |\n    |   |\' | |[_]| |  |\'  |\n    |   |  |_|###|_|  |   |\n    \'---\'--\'-/___\\-\'--\'---\'\n    ')

def print_end():
    if False:
        for i in range(10):
            print('nop')
    'Print the farewell to the program.'
    print('\n    ⢦⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⡤\n    ⠘⣿⣿⣿⣷⣦⣄⣀⠀⢠⠔⠀⢀⡼⠿⠿⢆⠀⠀⠲⣄⠀⣀⣠⣴⣾⣿⣿⣿⠇\n    ⠀⠈⠉⠉⠛⠛⠻⠿⢿⣿⠀⢀⣾⣷⡀⢀⣾⣷⡀⠀⣿⡿⠿⠿⠛⠛⠉⠉⠁⠀\n    ⠀⠀⣤⣤⣶⣶⣶⣶⣶⣿⣆⠈⠉⠉⠉⠉⠉⠉⠉⢠⣿⣶⣶⣶⣶⣶⣤⣤⠀⠀\n    ⠀⠀⠘⣿⡿⠟⠛⠉⣡⣿⣿⣷⣤⠀⢠⣆⠀⣤⣶⣿⣿⣬⡉⠛⠻⠿⣿⠇⠀⠀\n    ⠀⠀⠀⠀⠀⢀⣴⣿⡿⢋⣿⣿⠛⢠⣿⣿⡄⠛⢿⣿⡘⢿⣿⣦⣀⠀⠀⠀⠀⠀\n    ⠀⠀⠀⠀⠀⠉⠻⠏⠀⣸⣿⡇⢀⠻⣿⣿⠟⣀⠸⣿⣇⠀⠙⠟⠋⠀⠀⠀⠀⠀\n    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢠⡟⠁⣿⣿⠀⠻⣆⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n    ⠀⠀⠀⠀⠀⠀⠀⠀⠘⢟⠉⠙⠓⠀⠘⠏⠀⠘⠟⠉⡻⠋⠀⠀⠀⠀⠀⠀⠀⠀\n    ')
    print('')
    print('Take care of Hyrule!')
    print('')

def main():
    if False:
        for i in range(10):
            print('nop')
    'Allows the user to print one or more Triforce.'
    finish = False
    print_title()
    while finish == False:
        sizeTriforce = what_size()
        print_Triforce(sizeTriforce, sizeTriforce * 2 - 1)
        validateOption = False
        while validateOption == False:
            print('Do you want to generate another Triforce?')
            print('1 - Yes')
            print('2 - No')
            option = str(input(' '))
            if option == '2' or option == 'n' or option == 'N' or (option == 'No'):
                finish = True
                validateOption = True
                print('')
            elif option == '1' or option == 'y' or option == 'Y' or (option == 'Yes'):
                print('')
                validateOption = True
            else:
                print('Please! write a correct option.')
    print_end()
if __name__ == '__main__':
    main()