import PySimpleGUI as sg
'\n    Columns with a hard coded size that can have elements justified within it.\n    \n    The Column element can have the size set to a fixed size, but when doing so, PySimpleGUI has\n        a limitation that the contents can\'t be justified using the normal element_justification parameter.\n        \n    What to do?\n    \n    The Sizer Element to the rescue.\n    \n    PySimpleGUI likes to have layouts that size themselves rather than hard coded using a size parameter. The\n    Sizer Element enables you to create columns with fixed size by making the contents of your column a fixed size.\n    It is an invisible "padding" type of element.  It has a width and a height parameter.\n\n\n    Copyright 2021 PySimpleGUI\n'
'\nM#"""""""\'M                    dP                         \n##  mmmm. `M                   88                         \n#\'        .M 88d888b. .d8888b. 88  .dP  .d8888b. 88d888b. \nM#  MMMb.\'YM 88\'  `88 88\'  `88 88888"   88ooood8 88\'  `88 \nM#  MMMM\'  M 88       88.  .88 88  `8b. 88.  ... 88    88 \nM#       .;M dP       `88888P\' dP   `YP `88888P\' dP    dP \nM#########M\n'
col_interior = [[sg.Text('My Window')], [sg.In()], [sg.In()], [sg.Button('Go'), sg.Button('Exit'), sg.Cancel(), sg.Ok()]]
layout = [[sg.Text('This layout is broken.  The size of the Column is correct, but the elements are not justified')], [sg.Column(col_interior, element_justification='c', size=(500, 300), background_color='red')]]
window = sg.Window('Window Title', layout)
window.read(close=True)
'\nM""MMM""MMM""M                   dP                \nM  MMM  MMM  M                   88                \nM  MMP  MMP  M .d8888b. 88d888b. 88  .dP  .d8888b. \nM  MM\'  MM\' .M 88\'  `88 88\'  `88 88888"   Y8ooooo. \nM  `\' . \'\' .MM 88.  .88 88       88  `8b.       88 \nM    .d  .dMMM `88888P\' dP       dP   `YP `88888P\' \nMMMMMMMMMMMMMM\n'

def ColumnFixedSize(layout, size=(None, None), *args, **kwargs):
    if False:
        print('Hello World!')
    return sg.Column([[sg.Column([[sg.Sizer(0, size[1] - 1), sg.Column([[sg.Sizer(size[0] - 2, 0)]] + layout, *args, **kwargs, pad=(0, 0))]], *args, **kwargs)]], pad=(0, 0))
col_interior = [[sg.Text('My Window')], [sg.In()], [sg.In()], [sg.Button('Go'), sg.Button('Exit'), sg.Cancel(), sg.Ok()]]
layout = [[sg.Text('Below is a column that is 500 x 300')], [sg.Text('With the interior centered')], [ColumnFixedSize(col_interior, size=(500, 300), background_color='red', element_justification='c', vertical_alignment='t')]]
window = sg.Window('Window Title', layout)
window.read(close=True)