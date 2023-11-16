import time
functions = {'today': lambda x: time.strftime('%d/%m/%Y', time.localtime()).decode('latin1')}

def print_fnc(fnc, arg):
    if False:
        while True:
            i = 10
    if fnc in functions:
        return functions[fnc](arg)
    return ''