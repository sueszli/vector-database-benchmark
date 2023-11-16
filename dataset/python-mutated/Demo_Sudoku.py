import PySimpleGUI as sg, random
import numpy as np
from typing import List, Any, Union, Tuple, Dict
'\n    Sudoku Puzzle Demo\n\n    How to easily generate a GUI for a Sudoku puzzle.\n    The Window definition and creation is a single line of code.\n\n    Code to generate a playable puzzle was supplied from:\n    https://github.com/MorvanZhou/sudoku\n\n    Copyright 2020 PySimpleGUI.com\n\n'

def generate_sudoku(mask_rate):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Sukoku board\n\n    :param mask_rate: % of squares to hide\n    :type mask_rate: float\n    :rtype: List[numpy.ndarry, numpy.ndarry]\n    '
    while True:
        n = 9
        solution = np.zeros((n, n), np.int_)
        rg = np.arange(1, n + 1)
        solution[0, :] = np.random.choice(rg, n, replace=False)
        try:
            for r in range(1, n):
                for c in range(n):
                    col_rest = np.setdiff1d(rg, solution[:r, c])
                    row_rest = np.setdiff1d(rg, solution[r, :c])
                    avb1 = np.intersect1d(col_rest, row_rest)
                    (sub_r, sub_c) = (r // 3, c // 3)
                    avb2 = np.setdiff1d(np.arange(0, n + 1), solution[sub_r * 3:(sub_r + 1) * 3, sub_c * 3:(sub_c + 1) * 3].ravel())
                    avb = np.intersect1d(avb1, avb2)
                    solution[r, c] = np.random.choice(avb, size=1)[0]
            break
        except ValueError:
            pass
    puzzle = solution.copy()
    puzzle[np.random.choice([True, False], size=solution.shape, p=[mask_rate, 1 - mask_rate])] = 0
    return (puzzle, solution)

def check_progress(window, solution):
    if False:
        return 10
    "\n    Gives you a visual hint on your progress.\n    Red - You've got an incorrect number at the location\n    Yellow - You're missing an anwer for that location\n\n    :param window: The GUI's Window\n    :type window: sg.Window\n    :param solution: A 2D array containing the solution\n    :type solution: numpy.ndarray\n    :return: True if the puzzle has been solved correctly\n    :rtype: bool\n    "
    solved = True
    for (r, row) in enumerate(solution):
        for (c, col) in enumerate(row):
            value = window[r, c].get()
            if value:
                try:
                    value = int(value)
                except:
                    value = 0
                if value != solution[r][c]:
                    window[r, c].update(background_color='red')
                    solved = False
                else:
                    window[r, c].update(background_color=sg.theme_input_background_color())
            else:
                solved = False
                window[r, c].update(background_color='yellow')
    return solved

def main(mask_rate=0.7):
    if False:
        for i in range(10):
            print('nop')
    '"\n    The Main GUI - It does it all.\n\n    The "Board" is a grid that\'s 9 x 9.  Even though the layout is a grid of 9 Frames, the\n    addressing of the individual squares is via a key that\'s a tuple (0,0) to (8,8)\n    '

    def create_and_show_puzzle():
        if False:
            for i in range(10):
                print('nop')
        rate = mask_rate
        if window['-RATE-'].get():
            try:
                rate = float(window['-RATE-'].get())
            except:
                pass
        (puzzle, solution) = generate_sudoku(mask_rate=rate)
        for (r, row) in enumerate(puzzle):
            for (c, col) in enumerate(row):
                window[r, c].update(puzzle[r][c] if puzzle[r][c] else '', background_color=sg.theme_input_background_color())
        return (puzzle, solution)
    window = sg.Window('Sudoku', [[sg.Frame('', [[sg.I(random.randint(1, 9), justification='r', size=(3, 1), key=(fr * 3 + r, fc * 3 + c)) for c in range(3)] for r in range(3)]) for fc in range(3)] for fr in range(3)] + [[sg.B('Solve'), sg.B('Check'), sg.B('Hint'), sg.B('New Game')], [sg.T('Mask rate (0-1)'), sg.In(str(mask_rate), size=(3, 1), key='-RATE-')]], finalize=True)
    (puzzle, solution) = create_and_show_puzzle()
    while True:
        (event, values) = window.read()
        if event is None:
            break
        if event == 'Solve':
            for (r, row) in enumerate(solution):
                for (c, col) in enumerate(row):
                    window[r, c].update(solution[r][c], background_color=sg.theme_input_background_color())
        elif event == 'Check':
            solved = check_progress(window, solution)
            if solved:
                sg.popup('Solved! You have solved the puzzle correctly.')
        elif event == 'Hint':
            elem = window.find_element_with_focus()
            try:
                elem.update(solution[elem.Key[0]][elem.Key[1]], background_color=sg.theme_input_background_color())
            except:
                pass
        elif event == 'New Game':
            (puzzle, solution) = create_and_show_puzzle()
    window.close()
if __name__ == '__main__':
    mask_rate = 0.7
    main(mask_rate)