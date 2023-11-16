import PySimpleGUI as sg
'\n    Demo of using the vertical layout parameters and layout helper functions.\n    Three methods of vertical alignment are shown:\n    1. Using Column element to align a single element\n    2. Using vtop layout helper function to align a single element\n    3. Using vtop layout helper function to align an entire row\n\n    There is also a funciton provided that will convert an entire layout into \n    a top aligned layout.\n    \n    Copyright 2020 PySimpleGUI.org    \n'

def top_align_layout(layout):
    if False:
        i = 10
        return i + 15
    '\n    Given a layout, return a layout with all rows vertically adjusted to the top\n\n    :param layout: List[List[sg.Element]] The layout to justify\n    :return: List[List[sg.Element]]  The new layout that is all top justified\n    '
    new_layout = []
    for row in layout:
        new_layout.append(sg.vtop(row))
    return new_layout

def main():
    if False:
        print('Hello World!')
    layout = [[sg.T('This layout uses no vertical alignment. The default is "center"')], [sg.Text('On row 1'), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 1')], [sg.Button('OK')]]
    sg.Window('Example 1', layout).read(close=True)
    layout = [[sg.T('This uses a Column Element to align 1 element')], [sg.Col([[sg.Text('On row 1')]], vertical_alignment='top', pad=(0, 0)), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 1')], [sg.Button('OK')]]
    sg.Window('Example 2', layout).read(close=True)
    layout = [[sg.T('This layout uses the "vtop" layout helper function on 1 element')], [sg.vtop(sg.Text('On row 1')), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 1')], [sg.Button('OK')]]
    sg.Window('Example 3', layout).read(close=True)
    layout = [[sg.T('This layout uses the "vtop" layout helper function on 1 row')], sg.vtop([sg.Text('On row 1'), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 1')]), [sg.Button('OK')]]
    sg.Window('Example 4', layout).read(close=True)
    layout = [[sg.T('This layout uses the "vtop" for first part of row')], sg.vtop([sg.Text('On row 1'), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 1')]) + [sg.Text('More elements'), sg.CB('Last')], [sg.Button('OK')]]
    sg.Window('Example 5', layout).read(close=True)
    try:
        layout = [[sg.T('This layout uses the "vtop" for first part of row')], [*sg.vtop([sg.Text('On row 1'), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 1')]), sg.Text('More elements'), sg.CB('Last')], [sg.Button('OK')]]
        sg.Window('Example 5B', layout).read(close=True)
    except:
        print('Your version of Python likely does not support unpacking inside of a list')
    layout = [[sg.T('This layout has all rows top aligned using function')], [sg.Text('On row 1'), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 1')], [sg.Text('On row 2'), sg.Listbox(list(range(10)), size=(5, 4)), sg.Text('On row 2')], [sg.Button('OK')]]
    layout = top_align_layout(layout)
    sg.Window('Example 6', layout).read(close=True)
if __name__ == '__main__':
    main()