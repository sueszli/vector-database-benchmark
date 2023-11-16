import PySimpleGUI as sg
'\n    Demo - Add and "Delete" Rows from a window\n\n    This is cut-down version of the Fed-Ex package tracking demo\n\n    The purpose is to show a technique for making windows that grow by clicking an "Add Row" button\n    Each row can be individually "deleted".\n\n    The reason for using the quotes are "deleted" is that the elements are simply hidden.  The effect is the same as deleting them.\n\n    Copyright 2022 PySimpleGUI\n'

def item_row(item_num):
    if False:
        i = 10
        return i + 15
    '\n    A "Row" in this case is a Button with an "X", an Input element and a Text element showing the current counter\n    :param item_num: The number to use in the tuple for each element\n    :type:           int\n    :return:         List\n    '
    row = [sg.pin(sg.Col([[sg.B(sg.SYMBOL_X, border_width=0, button_color=(sg.theme_text_color(), sg.theme_background_color()), k=('-DEL-', item_num), tooltip='Delete this item'), sg.In(size=(20, 1), k=('-DESC-', item_num)), sg.T(f'Key number {item_num}', k=('-STATUS-', item_num))]], k=('-ROW-', item_num)))]
    return row

def make_window():
    if False:
        for i in range(10):
            print('nop')
    layout = [[sg.Text('Add and "Delete" Rows From a Window', font='_ 15')], [sg.Col([item_row(0)], k='-TRACKING SECTION-')], [sg.pin(sg.Text(size=(35, 1), font='_ 8', k='-REFRESHED-'))], [sg.T(sg.SYMBOL_X, enable_events=True, k='Exit', tooltip='Exit Application'), sg.T('↻', enable_events=True, k='Refresh', tooltip='Save Changes & Refresh'), sg.T('+', enable_events=True, k='Add Item', tooltip='Add Another Item')]]
    right_click_menu = [[''], ['Add Item', 'Edit Me', 'Version']]
    window = sg.Window('Window Title', layout, right_click_menu=right_click_menu, use_default_focus=False, font='_ 15', metadata=0)
    return window

def main():
    if False:
        print('Hello World!')
    window = make_window()
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Add Item':
            window.metadata += 1
            window.extend_layout(window['-TRACKING SECTION-'], [item_row(window.metadata)])
        elif event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Version':
            sg.popup_scrolled(__file__, sg.get_versions(), location=window.current_location(), keep_on_top=True, non_blocking=True)
        elif event[0] == '-DEL-':
            window['-ROW-', event[1]].update(visible=False)
    window.close()
if __name__ == '__main__':
    main()