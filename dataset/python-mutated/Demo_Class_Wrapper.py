import PySimpleGUI as sg
'\n    Demo - Class wrapper\n    \n    Using a class to encapsulate PySimpleGUI Window creation & event loop\n    \n    This is NOT a recommended design pattern.  It mimics the object oriented design that many OO-based\n    GUI frameworks use, but there is no advantage to structuring you code in his manner.  It adds\n    confusion, not clarity.  \n    \n    The class version is 18 lines of code.  The plain version is 13 lines of code.  \n    \n    Two things about the class wrapper jump out as adding confusion:\n    1. Unneccessary fragmentation of the event loop - the button click code is pulled out of the loop entirely\n    2. "self" clutters the code without adding value\n    \n\n    Copyright 2022, 2023 PySimpleGUI\n'
'\n    MM\'""""\'YMM dP                            \n    M\' .mmm. `M 88                            \n    M  MMMMMooM 88 .d8888b. .d8888b. .d8888b. \n    M  MMMMMMMM 88 88\'  `88 Y8ooooo. Y8ooooo. \n    M. `MMM\' .M 88 88.  .88       88       88 \n    MM.     .dM dP `88888P8 `88888P\' `88888P\' \n    MMMMMMMMMMM                               \n                                              \n    M""MMMMM""M                            oo                   \n    M  MMMMM  M                                                 \n    M  MMMMP  M .d8888b. 88d888b. .d8888b. dP .d8888b. 88d888b. \n    M  MMMM\' .M 88ooood8 88\'  `88 Y8ooooo. 88 88\'  `88 88\'  `88 \n    M  MMP\' .MM 88.  ... 88             88 88 88.  .88 88    88 \n    M     .dMMM `88888P\' dP       `88888P\' dP `88888P\' dP    dP \n    MMMMMMMMMMM\n'

class SampleGUI:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.layout = [[sg.Text('My layout')], [sg.Input(key='-IN-')], [sg.Button('Go'), sg.Button('Exit')]]
        self.window = sg.Window('My new window', self.layout)

    def run(self):
        if False:
            print('Hello World!')
        while True:
            (self.event, self.values) = self.window.read()
            if self.event in (sg.WIN_CLOSED, 'Exit'):
                break
            if self.event == 'Go':
                self.button_go()
        self.window.close()

    def button_go(self):
        if False:
            i = 10
            return i + 15
        sg.popup('Go button clicked', 'Input value:', self.values['-IN-'])
my_gui = SampleGUI()
my_gui.run()
'\n    M"""""""`YM                                       dP \n    M  mmmm.  M                                       88 \n    M  MMMMM  M .d8888b. 88d888b. 88d8b.d8b. .d8888b. 88 \n    M  MMMMM  M 88\'  `88 88\'  `88 88\'`88\'`88 88\'  `88 88 \n    M  MMMMM  M 88.  .88 88       88  88  88 88.  .88 88 \n    M  MMMMM  M `88888P\' dP       dP  dP  dP `88888P8 dP \n    MMMMMMMMMMM                                          \n                                                         \n    M""MMMMM""M                            oo                   \n    M  MMMMM  M                                                 \n    M  MMMMP  M .d8888b. 88d888b. .d8888b. dP .d8888b. 88d888b. \n    M  MMMM\' .M 88ooood8 88\'  `88 Y8ooooo. 88 88\'  `88 88\'  `88 \n    M  MMP\' .MM 88.  ... 88             88 88 88.  .88 88    88 \n    M     .dMMM `88888P\' dP       `88888P\' dP `88888P\' dP    dP \n    MMMMMMMMMMM\n'

def gui_function():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('My layout')], [sg.Input(key='-IN-')], [sg.Button('Go'), sg.Button('Exit')]]
    window = sg.Window('My new window', layout)
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'Go':
            sg.popup('Go button clicked', 'Input value:', values['-IN-'])
    window.close()
gui_function()