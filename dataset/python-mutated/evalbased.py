"""Unpacker for eval() based packers: runs JS code and returns result.
Works only if a JS interpreter (e.g. Mozilla's Rhino) is installed and
properly set up on host."""
from subprocess import PIPE, Popen
PRIORITY = 3

def detect(source):
    if False:
        for i in range(10):
            print('nop')
    'Detects if source is likely to be eval() packed.'
    return source.strip().lower().startswith('eval(function(')

def unpack(source):
    if False:
        while True:
            i = 10
    'Runs source and return resulting code.'
    return jseval('print %s;' % source[4:]) if detect(source) else source

def jseval(script):
    if False:
        while True:
            i = 10
    'Run code in the JS interpreter and return output.'
    try:
        interpreter = Popen(['js'], stdin=PIPE, stdout=PIPE)
    except OSError:
        return script
    (result, errors) = interpreter.communicate(script)
    if interpreter.poll() or errors:
        return script
    return result