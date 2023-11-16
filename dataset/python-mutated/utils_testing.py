import os
from urh import settings

def trace_calls(frame, event, arg):
    if False:
        while True:
            i = 10
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    caller = frame.f_back
    caller_line_no = caller.f_lineno
    caller_filename = caller.f_code.co_filename
    if 'urh' in caller_filename or 'urh' in func_filename:
        if 'logging' in caller_filename or 'logging' in func_filename:
            return
        if '_test' in caller_filename or '_test' in func_filename:
            start = '\x1b[91m'
        else:
            start = '\x1b[0;32m'
        end = '\x1b[0;0m'
    else:
        (start, end) = ('', '')
    print('%s Call to %s on line %s of %s from line %s of %s %s' % (start, func_name, func_line_no, func_filename, caller_line_no, caller_filename, end))
    return
global settings_written

def write_settings():
    if False:
        for i in range(10):
            print('nop')
    global settings_written
    try:
        settings_written
    except NameError:
        settings_written = True
        settings.write('not_show_close_dialog', True)
        settings.write('not_show_save_dialog', True)
        settings.write('NetworkSDRInterface', True)
        settings.write('align_labels', True)
f = os.readlink(__file__) if os.path.islink(__file__) else __file__
path = os.path.realpath(os.path.join(f, '..'))

def get_path_for_data_file(filename):
    if False:
        print('Hello World!')
    return os.path.join(path, 'data', filename)