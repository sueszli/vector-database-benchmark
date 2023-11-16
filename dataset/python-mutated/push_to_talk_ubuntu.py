import keyboard
import subprocess
is_muted = False

def unmute():
    if False:
        print('Hello World!')
    global is_muted
    if not is_muted:
        return
    is_muted = False
    subprocess.call('amixer set Capture cap', shell=True)

def mute():
    if False:
        print('Hello World!')
    global is_muted
    is_muted = True
    subprocess.call('amixer set Capture nocap', shell=True)
if __name__ == '__main__':
    is_muted = True
    mute()
    keyboard.add_hotkey('win', unmute)
    keyboard.add_hotkey('win', mute, trigger_on_release=True)
    keyboard.wait()