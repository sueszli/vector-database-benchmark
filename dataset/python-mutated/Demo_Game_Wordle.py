import PySimpleGUI as sg
import copy
'\n    Wordle GUI Demo\n\n    Enter characters for each position\n    Press enter or click enter to submit a row\n\n    This is a prototype GUI of the front-end for WORDL\n    It currently:\n        * Takes input\n            * Makes sure only characters\n            * Automatically converts to upper case\n            * Handles backspace\n            * Checks for Enter key\n        * Compares against a word (the constant "answer")\n        * Color codes the submitted guess\n\n    To complete an application, you\'ll need to:\n        * Supply a word to guess from list of words\n        * Check if user\'s submission is a word (I think this is how WORDLE works)\n\n    Copyright 2022 PySimpleGUI\n'
answer = 'WORDS'

def TextChar(value, key):
    if False:
        print('Hello World!')
    return sg.Input(value, key=key, font='Courier 22', size=(1, 1), disabled_readonly_background_color='gray', border_width=1, p=1, enable_events=True, disabled=True)

def main():
    if False:
        while True:
            i = 10
    layout = [[sg.Text('Wordle', font='_ 20')], [[TextChar('', (row, col)) for col in range(5)] for row in range(6)], [sg.B('Enter', bind_return_key=True)], [sg.Text('Or press enter', font='_ 10')]]
    window = sg.Window('Wordle', layout, finalize=True, element_justification='c')
    (cur_row, correct) = (0, False)
    [window[cur_row, col].update(disabled=False) for col in range(5)]
    window.bind('<BackSpace>', '-BACKSPACE-')
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED:
            break
        if isinstance(event, tuple):
            if len(values[event]):
                (row, col) = event
                char_input = values[event][-1]
                if not char_input.isalpha():
                    window[event].update('')
                else:
                    window[event].update(char_input.upper()[0])
                    if col < 4:
                        window[row, col + 1].set_focus()
        elif event == 'Enter' and cur_row < 5:
            guess = ''.join([values[cur_row, j] for j in range(5)])
            answer2 = copy.copy(answer)
            for (i, letter) in enumerate(guess):
                if letter == answer2[i]:
                    window[cur_row, i].update(background_color='green', text_color='white')
                    answer2 = answer2.replace(letter, '*')
                elif letter in answer2:
                    window[cur_row, i].update(background_color='#C9B359', text_color='white')
                    answer2 = answer2.replace(letter, '*')
                else:
                    window[cur_row, i].update(background_color='gray', text_color='white')
            if guess == answer:
                correct = True
                break
            cur_row += 1
            [window[cur_row, col].update(disabled=False) for col in range(5)]
            window[cur_row, 0].set_focus()
        elif event == 'Enter' and cur_row == 5:
            correct = False
            break
        elif event == '-BACKSPACE-':
            current_focus = window.find_element_with_focus()
            current_key = current_focus.Key
            if isinstance(current_key, tuple):
                window[current_key].update('')
                if current_key[1] > 0:
                    window[current_key[0], current_key[1] - 1].set_focus()
                    window[current_key[0], current_key[1] - 1].update('')
    if correct:
        sg.popup('You win!')
    else:
        sg.popup(f'Sorry... the answer was {answer}')
    window.close()
if __name__ == '__main__':
    main()