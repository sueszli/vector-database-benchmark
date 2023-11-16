import PySimpleGUI as sg
web = False
"\n    If you're using the PySimpleGUI color themes, then your code will a line that looks something like this:\n        sg.change_look_and_feel('Light Green 1') or sg.change_look_and_feel('LightGreen1')\n"
sg.preview_all_look_and_feel_themes()
WINDOW_BACKGROUND = 'lightblue'
web = False
sg.change_look_and_feel('Default')

def sample_layout():
    if False:
        while True:
            i = 10
    return [[sg.Text('Text element'), sg.InputText('Input data here', size=(15, 1))], [sg.Button('Ok'), sg.Button('Cancel'), sg.Slider((1, 10), orientation='h', size=(10, 15))]]
layout = [[sg.Text('List of Themes Provided by PySimpleGUI', font='Default 18', background_color=WINDOW_BACKGROUND)]]
FRAMES_PER_ROW = 9
names = sg.list_of_look_and_feel_values()
names.sort()
row = []
for (count, theme) in enumerate(names):
    sg.change_look_and_feel(theme)
    if not count % FRAMES_PER_ROW:
        layout += [row]
        row = []
    row += [sg.Frame(theme, sample_layout() if not web else [[sg.T(theme)]] + sample_layout())]
if row:
    layout += [row]
window = sg.Window('Custom Preview of Themes', layout, background_color=WINDOW_BACKGROUND)
window.read()
window.close()
del window