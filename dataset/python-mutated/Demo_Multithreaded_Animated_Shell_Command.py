import subprocess
import PySimpleGUI as sg
import threading
'\n    Demo - Run a shell command while displaying an animated GIF to inform the user the \n    program is still running.\n    If you have a GUI and you start a subprocess to run a shell command, the GUI essentually\n    locks up and often the operation system will off to terminate the program for you.\n\n    This demo fixes this situation by running the subprocess as a Thread.   This enables\n    the subproces to run async to the main program.  The main program then simply runs a loop,\n    waiting for the thread to complete running. \n\n    The output from the subprocess is saved and displayed in a scrolled popup.\n'

def process_thread():
    if False:
        print('Hello World!')
    global proc
    proc = subprocess.run('pip list', shell=True, stdout=subprocess.PIPE)

def main():
    if False:
        return 10
    thread = threading.Thread(target=process_thread, daemon=True)
    thread.start()
    while True:
        sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, 'Loading list of packages', time_between_frames=100)
        thread.join(timeout=0.1)
        if not thread.is_alive():
            break
    sg.popup_animated(None)
    output = proc.__str__().replace('\\r\\n', '\n')
    sg.popup_scrolled(output, font='Courier 10')
if __name__ == '__main__':
    main()