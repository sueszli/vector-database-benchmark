import PySimpleGUI as sg
"\n   Use this code as a starting point for creating your own Popup functions.  \n   Rather than creating a long list of Popup high-level API calls, PySimpleGUI provides\n   you with the tools to easily create your own.  If you need more than what the standard popup_get_text and\n   other calls provide, then it's time for you to graduate into making your own windows.  Or, maybe you need\n   another window that pops-up over your primary window.  Whatever the need, don't hesitate to dive in\n   and create your own Popup call.\n   \n   This example is for a DropDown / Combobox Popup.  You provide it with a title, a message and the list\n   of values to choose from. It mimics the return values of existing Popup calls (None if nothing was input)\n"

def PopupDropDown(title, text, values):
    if False:
        i = 10
        return i + 15
    window = sg.Window(title, [[sg.Text(text)], [sg.DropDown(values, key='-DROP-')], [sg.OK(), sg.Cancel()]])
    (event, values) = window.read()
    return None if event != 'OK' else values['-DROP-']
values = ['choice {}'.format(x) for x in range(30)]
print(PopupDropDown('My Title', 'Please make a selection', values))