import subprocess
import sys

def _get_meipass_value():
    if False:
        while True:
            i = 10
    if sys.platform.startswith('win'):
        command = 'echo %_MEIPASS2%'
    else:
        command = 'echo $_MEIPASS2'
    stdout = subprocess.check_output(command, shell=True)
    meipass = stdout.strip()
    if meipass.startswith(b'%'):
        meipass = ''
    return meipass
meipass = _get_meipass_value()
print(meipass)
print('_MEIPASS2 value: %s' % sys._MEIPASS)
if meipass:
    raise SystemExit('Error: _MEIPASS2 env variable available in subprocess.')