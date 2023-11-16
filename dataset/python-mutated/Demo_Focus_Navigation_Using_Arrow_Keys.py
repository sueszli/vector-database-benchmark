import PySimpleGUI as sg
"\n    Demo - Navigating a window's focus using arrow keys\n    \n    This Demo Program has 2 features of PySimpleGUI in use:\n    1. Binding the arrow keys\n    2. Navigating a window's elements using focus\n    \n    The first step is to bind the left, right and down arrows to an event.\n    The call to window.bind will cause events to be generated when these keys are pressed\n    \n    The next step is to add the focus navigation to your event loop.\n    When the right key is pressed, the focus moves to the element that should get focus next\n    When the left arrow key is pressed, the focus moves to the previous element\n    And when the down arrow is pressed the program exits\n\n\n    Copyright 2022 PySimpleGUI\n"

def main():
    if False:
        for i in range(10):
            print('nop')
    layout = [[sg.Text('My Window')], [sg.Input(key='-IN-')], [sg.Input(key='-IN2-')], [sg.Input(key='-IN3-')], [sg.Input(key='-IN4-')], [sg.Input(key='-IN5-')], [sg.Input(key='-IN6-')], [sg.Input(key='-IN7-')], [sg.Button('Go'), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout, finalize=True)
    window.bind('<Right>', '-NEXT-')
    window.bind('<Left>', '-PREV-')
    window.bind('<Down>', 'Exit')
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-NEXT-':
            next_element = window.find_element_with_focus().get_next_focus()
            next_element.set_focus()
        if event == '-PREV-':
            prev_element = window.find_element_with_focus().get_previous_focus()
            prev_element.set_focus()
    window.close()
if __name__ == '__main__':
    main()