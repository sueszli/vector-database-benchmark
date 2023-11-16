import PySimpleGUI as sg
'\n    Demo - Listbox Using Objects\n\n    Several elements can take not just strings, but objects.  The Listsbox is one of them.\n    This demo show how you can use objects directly in a Listbox in a way that you can access\n        information about each object that is different than what is shown in the Window.\n\n    The important part of this design pattern is the use of the __str__ method in your item objects.\n        This method is what determines what is shown in the window.\n\n    Copyright 2022 PySimpleGUI\n'

class Item:

    def __init__(self, internal, shown):
        if False:
            for i in range(10):
                print('nop')
        self.internal = internal
        self.shown = shown

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.shown
my_item_list = [Item(f'Internal {i}', f'shown {i}') for i in range(100)]
layout = [[sg.Text('Select 1 or more items and click "Go"')], [sg.Listbox(my_item_list, key='-LB-', s=(20, 20), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)], [sg.Output(s=(40, 10))], [sg.Button('Go'), sg.Button('Exit')]]
window = sg.Window('Listbox Using Objects', layout)
while True:
    (event, values) = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Go':
        print('You selected:')
        for item in values['-LB-']:
            print(item.internal)
window.close()