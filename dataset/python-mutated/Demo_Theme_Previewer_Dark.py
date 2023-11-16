import PySimpleGUI as sg
'\n    If you\'re using the PySimpleGUI color themes, then your code will a line that looks something like this:\n        sg.theme(\'Light Green 1\') or sg.theme(\'LightGreen1\')\n        \n    This demo shows how to access the list of all "dark themes" as an example of how you can build your own previewer\n    \n    Copyright 2020 PySimpleGUI.org\n'
window_background = 'black'

def sample_layout():
    if False:
        while True:
            i = 10
    "\n    Creates a small window that will represent the colors of the theme. This is an individual theme's preview\n    :return: layout of a little preview window\n    :rtype: List[List[Element]]\n    "
    return [[sg.Text('Text element'), sg.InputText('Input data here', size=(15, 1))], [sg.Button('Ok'), sg.Button('Cancel'), sg.Slider((1, 10), orientation='h', size=(10, 15))]]
layout = [[sg.Text('List of Dark Themes Provided by PySimpleGUI', font='Default 18', background_color=window_background)]]
FRAMES_PER_ROW = 9
names = [name for name in sg.theme_list() if 'dark' in name.lower()]
row = []
for (count, theme) in enumerate(names):
    sg.theme(theme)
    if not count % FRAMES_PER_ROW:
        layout += [row]
        row = []
    row += [sg.Frame(theme, sample_layout())]
if row:
    layout += [row]
sg.Window('Custom Preview of Themes', layout, background_color=window_background).read(close=True)