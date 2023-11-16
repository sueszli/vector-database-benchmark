import PySimpleGUI as sg
import time
import threading
'\n    Demo - Multi-threaded - Show Windows and perform other PySimpleGUI calls in what appread to be from a thread\n    \n    Just so that it\'s clear, you CANNOT make PySimpleGUI calls directly from a thread.  There is ONE exception to this\n    rule.  A thread may call  window.write_event_values which enables it to communicate to a window through the window.read calls.\n\n    The main GUI will not be visible on your screen nor on your taskbar despite running in the background.  The calls you \n    make, such as popup, or even Window.read will create windows that your user will see.\n    \n    The basic function that you\'ll use in your thread has this format:\n        make_delegate_call(lambda: sg.popup(\'This is a popup\', i, auto_close=True, auto_close_duration=2, keep_on_top=True, non_blocking=True))\n\n    Everything after the "lambda" looks exactly like a PySimpleGUI call.\n    If you want to display an entire window, then the suggestion is to put it into a function and pass the function to make_delegate_call\n    \n    Note - the behavior of variables may be a bit of a surprise as they are not evaluated until the mainthread processes the event.  This means\n    in the example below that the counter variable being passed to the popup will not appear to be counting correctly.  This is because the\n    value shown will be the value at the time the popup is DISPLAYED, not the value when the make_delegate_call was made. \n    \n    Copyright 2022 PySimpleGUI\n'
window: sg.Window = None

def the_thread():
    if False:
        print('Hello World!')
    '\n    This is code that is unique to your application.  It wants to "make calls to PySimpleGUI", but it cannot directly do so.\n    Instead it will send the request to make the call to the mainthread that is running the GUI.\n\n    :return:\n    '
    while window is None:
        time.sleep(0.2)
    for i in range(5):
        time.sleep(0.2)
        make_delegate_call(lambda : sg.popup('This is a popup', i, relative_location=(0, -300), auto_close=True, auto_close_duration=2, keep_on_top=True, non_blocking=True))
        make_delegate_call(lambda : sg.popup_scrolled(__file__, sg.get_versions(), auto_close=True, auto_close_duration=1.5, non_blocking=True))
    make_delegate_call(lambda : sg.popup('One last popup before exiting...', relative_location=(-200, -200)))
    window.write_event_value('-THREAD EXIT-', None)

def make_delegate_call(func):
    if False:
        for i in range(10):
            print('nop')
    "\n    Make a delegate call to PySimpleGUI.\n\n    :param func:    A lambda expression most likely.  It's a function that will be called by the mainthread that's executing the GUI\n    :return:\n    "
    if window is not None:
        window.write_event_value('-THREAD DELEGATE-', func)

def main():
    if False:
        while True:
            i = 10
    global window
    layout = [[sg.Text('', k='-T-')]]
    window = sg.Window('Invisible window', layout, no_titlebar=True, alpha_channel=0, finalize=True, font='_ 1', margins=(0, 0), element_padding=(0, 0))
    window.hide()
    while True:
        (event, values) = window.read()
        if event in ('Exit', sg.WIN_CLOSED):
            break
        if event == '-THREAD DELEGATE-':
            try:
                values[event]()
            except Exception as e:
                sg.popup_error_with_traceback('Error calling your function passed to GUI', event, values, e)
        elif event == '-THREAD EXIT-':
            break
    window.close()
if __name__ == '__main__':
    threading.Thread(target=the_thread, daemon=True).start()
    main()