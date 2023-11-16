import unittest
from unittest.mock import patch, call
import random

class Cell:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.visile = False

class Door(Cell):

    def __init__(self):
        if False:
            return 10
        self.visile = True

    def __str__(self):
        if False:
            return 10
        return '🚪'

    def enter(self):
        if False:
            i = 10
            return i + 15
        print('Pregunta')
        return False
questions_answers = [('¿Cuál es el animal terrestre más grande del mundo? ', 'elefante'), ('¿Qué animal es conocido como el "Rey de la Selva"? ', 'leon'), ('¿Cuál es el único mamífero capaz de volar? ', 'murcielago'), ('¿Qué animal es famoso por su franja negra y blanca y es originario de China? ', 'panda'), ('¿Qué animal es conocido por cambiar de color para camuflarse en su entorno? ', 'camaleon'), ('¿Cuál es el animal más rápido del mundo? ', 'guepardo'), ('¿Qué animal es famoso por regenerar sus extremidades si se cortan? ', 'estrella'), ('¿Cuál es el animal mamífero más grande del océano? ', 'ballena'), ('¿Cuál es el ave que no puede volar y es conocida por ser un excelente nadador? ', 'pingüino'), ('¿Qué animal es el símbolo de la longevidad en la cultura china? ', 'tortuga')]

def question():
    if False:
        print('Hello World!')
    (question, answer) = random.choice(questions_answers)
    while True:
        user_answer = input(question)
        if answer != user_answer:
            print('mal! trampa [' + answer + ']')
        else:
            break

class Enigma(Cell):

    def __str__(self):
        if False:
            while True:
                i = 10
        return '❓' if self.visile else '⬜️'

    def enter(self):
        if False:
            while True:
                i = 10
        print('Pregunta')
        question()
        self.visile = True
        return False

class Candy(Cell):

    def __str__(self):
        if False:
            while True:
                i = 10
        return '🍭' if self.visile else '⬜️'

    def enter(self):
        if False:
            for i in range(10):
                print('nop')
        print('Candy!!')
        self.visile = True
        return True

class Ghost(Cell):

    def __str__(self):
        if False:
            while True:
                i = 10
        return '👻' if self.visile else '⬜️'

    def enter(self):
        if False:
            print('Hello World!')
        print('Fantasma, 2 Preguntas!')
        question()
        question()
        self.visile = True
        return False

class House:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_special_cells()
        self.cells = [[self.get_cells(row_index, col_index) for col_index in range(4)] for row_index in range(4)]

    def __str__(self):
        if False:
            return 10
        output = ''
        for (row_index, row) in enumerate(self.cells):
            for (col_index, item) in enumerate(row):
                if row_index == self.player_row and col_index == self.player_col:
                    output += '😐'
                else:
                    output += str(item)
            output += '\n'
        return output

    def get_cells(self, row_index, col_index):
        if False:
            return 10
        if (row_index, col_index) in self.special_cells:
            return self.special_cells[row_index, col_index]
        elif random.randint(1, 10) == 1:
            return Ghost()
        else:
            return Enigma()

    def setup_special_cells(self):
        if False:
            i = 10
            return i + 15
        door_pos_row = random.randint(0, 3)
        door_pos_col = random.randint(0, 3)
        self.special_cells = {(door_pos_row, door_pos_col): Door()}
        self.player_row = door_pos_row
        self.player_col = door_pos_col
        candy_pos_row = random.randint(0, 3)
        candy_pos_col = random.randint(0, 3)
        while (candy_pos_row, candy_pos_col) in self.special_cells:
            candy_pos_row = random.randint(0, 3)
            candy_pos_col = random.randint(0, 3)
        self.special_cells[candy_pos_row, candy_pos_col] = Candy()

    def move(self):
        if False:
            i = 10
            return i + 15
        while True:
            direction = input('en que direccion quiere ir (N/S/E/O)')
            if direction == 'N':
                if self.player_row == 0:
                    print('estas saliendo del tablero')
                else:
                    self.player_row -= 1
                    break
            if direction == 'S':
                if self.player_row == 3:
                    print('estas saliendo del tablero')
                else:
                    self.player_row += 1
                    break
            if direction == 'E':
                if self.player_col == 0:
                    print('estas saliendo del tablero')
                else:
                    self.player_col -= 1
                    break
            if direction == 'O':
                if self.player_col == 3:
                    print('estas saliendo del tablero')
                else:
                    self.player_col += 1
                    break

    def enter_cell(self):
        if False:
            while True:
                i = 10
        return self.cells[self.player_row][self.player_col].enter()

def main():
    if False:
        print('Hello World!')
    house = House()
    game_over = False
    while not game_over:
        print(house)
        house.move()
        game_over = house.enter_cell()

class TestDoor(unittest.TestCase):

    def test_init(self):
        if False:
            return 10
        door = Door()
        self.assertIsNotNone(door)
        self.assertTrue(door.visile)

    def test__str__(self):
        if False:
            while True:
                i = 10
        door = Door()
        self.assertEqual(str(door), '🚪')

class TestCandy(unittest.TestCase):

    def test_init(self):
        if False:
            return 10
        candy = Candy()
        self.assertIsNotNone(candy)
        self.assertFalse(candy.visile)

    def test__str__invisible(self):
        if False:
            return 10
        candy = Candy()
        self.assertEqual(str(candy), '⬜️')

    def test__str__visible(self):
        if False:
            i = 10
            return i + 15
        candy = Candy()
        candy.visile = True
        self.assertEqual(str(candy), '🍭')

class TestEnigma(unittest.TestCase):

    def test_init(self):
        if False:
            while True:
                i = 10
        engima = Enigma()
        self.assertIsNotNone(engima)
        self.assertFalse(engima.visile)

    def test__str__invisible(self):
        if False:
            for i in range(10):
                print('nop')
        engima = Enigma()
        self.assertEqual(str(engima), '⬜️')

    def test__str__visible(self):
        if False:
            for i in range(10):
                print('nop')
        engima = Enigma()
        engima.visile = True
        self.assertEqual(str(engima), '❓')

class TestGhost(unittest.TestCase):

    def test_init(self):
        if False:
            return 10
        ghost = Ghost()
        self.assertIsNotNone(ghost)
        self.assertFalse(ghost.visile)

    def test__str__invisible(self):
        if False:
            i = 10
            return i + 15
        ghost = Ghost()
        self.assertEqual(str(ghost), '⬜️')

    def test__str__visible(self):
        if False:
            return 10
        ghost = Ghost()
        ghost.visile = True
        self.assertEqual(str(ghost), '👻')

class TestCell(unittest.TestCase):

    def test_init(self):
        if False:
            return 10
        cell = Cell()
        self.assertIsNotNone(cell)
        self.assertFalse(cell.visile)

class TestHouse(unittest.TestCase):

    def test_init(self):
        if False:
            while True:
                i = 10
        house = House()
        self.assertEqual(len(house.cells), 4)
        for room_index in range(4):
            self.assertEqual(len(house.cells[room_index]), 4)

    @patch('random.randint', side_effect=[2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_door_pos(self, randint_patched):
        if False:
            for i in range(10):
                print('nop')
        house = House()
        self.assertIsInstance(house.cells[2][3], Door)

    @patch('random.randint', side_effect=[2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_character_pos(self, randint_patched):
        if False:
            for i in range(10):
                print('nop')
        house = House()
        self.assertEqual(house.player_row, 2)
        self.assertEqual(house.player_col, 3)

    @patch('random.randint', side_effect=[2, 3, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_candy_pos(self, randint_patched):
        if False:
            i = 10
            return i + 15
        house = House()
        self.assertIsInstance(house.cells[2][3], Door)
        self.assertIsInstance(house.cells[3][1], Candy)

    @patch('random.randint', side_effect=[2, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_ghosts_pos(self, randint_patched):
        if False:
            i = 10
            return i + 15
        house = House()
        self.assertIsInstance(house.cells[2][3], Door)
        self.assertIsInstance(house.cells[3][1], Candy)
        self.assertIsNone(house.cells[0][0])
        self.assertIsInstance(house.cells[0][1], Ghost)
        self.assertIsInstance(house.cells[3][3], Ghost)

    @patch('random.randint', side_effect=[2, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_ghosts_pos(self, randint_patched):
        if False:
            while True:
                i = 10
        house = House()
        self.assertEqual(str(house), '⬜️⬜️⬜️⬜️\n' + '⬜️⬜️⬜️⬜️\n' + '⬜️⬜️⬜️😐\n' + '⬜️⬜️⬜️⬜️\n')

    @patch('builtins.print')
    @patch('builtins.input', return_value='N')
    @patch('random.randint', side_effect=[2, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_move_N(self, randint_patched, input_patched, print_patched):
        if False:
            return 10
        house = House()
        house.move()
        self.assertEqual(house.player_row, 1)
        print_patched.assert_not_called()
        input_patched.assert_called_once()

    @patch('builtins.print')
    @patch('builtins.input', side_effect=['N', 'S'])
    @patch('random.randint', side_effect=[0, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_move_N_error(self, randint_patched, input_patched, print_patched):
        if False:
            return 10
        house = House()
        house.move()
        self.assertEqual(house.player_row, 1)
        self.assertEqual(input_patched.call_count, 2)
        self.assertEqual(print_patched.call_args_list, [call('estas saliendo del tablero')])

    @patch('builtins.print')
    @patch('builtins.input', return_value='S')
    @patch('random.randint', side_effect=[2, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_move_S(self, randint_patched, input_patched, print_patched):
        if False:
            while True:
                i = 10
        house = House()
        house.move()
        self.assertEqual(house.player_row, 3)
        print_patched.assert_not_called()
        input_patched.assert_called_once()

    @patch('builtins.print')
    @patch('builtins.input', side_effect=['S', 'N'])
    @patch('random.randint', side_effect=[3, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_move_S_error(self, randint_patched, input_patched, print_patched):
        if False:
            print('Hello World!')
        house = House()
        house.move()
        self.assertEqual(house.player_row, 2)
        self.assertEqual(input_patched.call_count, 2)
        self.assertEqual(print_patched.call_args_list, [call('estas saliendo del tablero')])

    @patch('builtins.print')
    @patch('builtins.input', return_value='E')
    @patch('random.randint', side_effect=[2, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_move_E(self, randint_patched, input_patched, print_patched):
        if False:
            for i in range(10):
                print('nop')
        house = House()
        house.move()
        self.assertEqual(house.player_col, 2)
        print_patched.assert_not_called()
        input_patched.assert_called_once()

    @patch('builtins.print')
    @patch('builtins.input', side_effect=['E', 'O'])
    @patch('random.randint', side_effect=[3, 0, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_move_E_error(self, randint_patched, input_patched, print_patched):
        if False:
            while True:
                i = 10
        house = House()
        house.move()
        self.assertEqual(house.player_col, 1)
        self.assertEqual(input_patched.call_count, 2)
        self.assertEqual(print_patched.call_args_list, [call('estas saliendo del tablero')])
if __name__ == '__main__':
    main()