import PySimpleGUI as sg
from time import sleep
sg.theme('Dark Blue 3')
'\n    Demonstration of simple and multiple one_line_progress_meter\'s as well as the Progress Meter Element\n    \n    There are 4 demos\n    1. Manually updated progress bar\n    2. Custom progress bar built into your window, updated in a loop\n    3. one_line_progress_meters, nested meters showing how 2 can be run at the same time.\n    4. An "iterable" style progress meter - a wrapper for one_line_progress_meters\n    \n    If the software determined that a meter should be cancelled early, \n        calling OneLineProgresMeterCancel(key) will cancel the meter with the matching key\n'
'\n    The simple case is that you want to add a single meter to your code.  The one-line solution.\n    This demo function shows 3 different one_line_progress_meter tests\n        1. A horizontal with red and white bar colors\n        2. A vertical bar with default colors\n        3. A test showing 2 running at the same time \n'

def demo_one_line_progress_meter():
    if False:
        print('Hello World!')
    for i in range(10000):
        if not sg.one_line_progress_meter('My 1-line progress meter', i + 1, 10000, 'meter key', 'MY MESSAGE1', 'MY MESSAGE 2', orientation='h', no_titlebar=True, grab_anywhere=True, bar_color=('white', 'red')):
            print('Hit the break')
            break
    for i in range(10000):
        if not sg.one_line_progress_meter('My 1-line progress meter', i + 1, 10000, 'meter key', 'MY MESSAGE1', 'MY MESSAGE 2', orientation='v'):
            print('Hit the break')
            break
    layout = [[sg.Text('One-Line Progress Meter Demo', font='Any 18')], [sg.Text('Outer Loop Count', size=(15, 1), justification='r'), sg.Input(default_text='100', size=(5, 1), key='CountOuter'), sg.Text('Delay'), sg.Input(default_text='10', key='TimeOuter', size=(5, 1)), sg.Text('ms')], [sg.Text('Inner Loop Count', size=(15, 1), justification='r'), sg.Input(default_text='100', size=(5, 1), key='CountInner'), sg.Text('Delay'), sg.Input(default_text='10', key='TimeInner', size=(5, 1)), sg.Text('ms')], [sg.Button('Show', pad=((0, 0), 3), bind_return_key=True), sg.Text('me the meters!')]]
    window = sg.Window('One-Line Progress Meter Demo', layout)
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == 'Show':
            max_outer = int(values['CountOuter'])
            max_inner = int(values['CountInner'])
            delay_inner = int(values['TimeInner'])
            delay_outer = int(values['TimeOuter'])
            for i in range(max_outer):
                if not sg.one_line_progress_meter('Outer Loop', i + 1, max_outer, 'outer'):
                    break
                sleep(delay_outer / 1000)
                for j in range(max_inner):
                    if not sg.one_line_progress_meter('Inner Loop', j + 1, max_inner, 'inner'):
                        break
                    sleep(delay_inner / 1000)
    window.close()
'\n    Manually Updated Test\n    Here is an example for when you want to "sprinkle" progress bar updates in multiple\n    places within your source code and you\'re not running an event loop.\n    Note that UpdateBar is special compared to other Update methods. It also refreshes\n    the containing window and checks for window closure events\n    The sleep calls are here only for demonstration purposes.  You should NOT be adding\n    these kinds of sleeps to a GUI based program normally.\n'

def manually_updated_meter_test():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('This meter is manually updated 4 times')], [sg.ProgressBar(max_value=10, orientation='h', size=(20, 20), key='progress')]]
    window = sg.Window('Custom Progress Meter', layout, finalize=True)
    progress_bar = window['progress']
    progress_bar.update_bar(1)
    sleep(2)
    progress_bar.update_bar(2)
    sleep(2)
    progress_bar.update_bar(6)
    sleep(2)
    progress_bar.update_bar(9)
    sleep(2)
    window.close()
'\n    This function shows how to create a custom window with a custom progress bar and then\n    how to update the bar to indicate progress is being made\n'

def custom_meter_example():
    if False:
        return 10
    layout = [[sg.Text('A typical custom progress meter')], [sg.ProgressBar(1, orientation='h', size=(20, 20), key='progress')], [sg.Cancel()]]
    window = sg.Window('Custom Progress Meter', layout)
    progress_bar = window['progress']
    for i in range(10000):
        (event, values) = window.read(timeout=0)
        if event == 'Cancel' or event == None:
            break
        progress_bar.update_bar(i + 1, 10000)
    window.close()
'\n    A Wrapper for one_line_progress_meter that combines your iterable with a progress meter into a single interface.  Two functions are provided  \n    progess_bar - The basic wrapper\n    progress_bar_range - A "convienence function" if you wanted to specify like a range\n'

def progress_bar(key, iterable, *args, title='', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Takes your iterable and adds a progress meter onto it\n    :param key: Progress Meter key\n    :param iterable: your iterable\n    :param args: To be shown in one line progress meter\n    :param title: Title shown in meter window\n    :param kwargs: Other arguments to pass to one_line_progress_meter\n    :return:\n    '
    sg.one_line_progress_meter(title, 0, len(iterable), key, *args, **kwargs)
    for (i, val) in enumerate(iterable):
        yield val
        if not sg.one_line_progress_meter(title, i + 1, len(iterable), key, *args, **kwargs):
            break

def progress_bar_range(key, start, stop=None, step=1, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Acts like the range() function but with a progress meter built-into it\n    :param key: progess meter's key\n    :param start: low end of the range\n    :param stop: Uppder end of range\n    :param step:\n    :param args:\n    :param kwargs:\n    :return:\n    "
    return progress_bar(key, range(start, stop, step), *args, **kwargs)

def demo_iterable_progress_bar():
    if False:
        print('Hello World!')
    my_list = list(range(1000))
    for value in progress_bar('bar1', my_list, title='First bar Test'):
        print(value)
    my_list = [x for x in progress_bar('bar1', my_list, title='Second bar Test')]
demo_iterable_progress_bar()
manually_updated_meter_test()
custom_meter_example()
demo_one_line_progress_meter()