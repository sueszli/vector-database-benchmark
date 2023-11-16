from manticore import Manticore

def fixme():
    if False:
        while True:
            i = 10
    raise Exception('Fill in the blanks!')
m = Manticore('multiple-styles')
m.concrete_data = 'infiltrate miami!'
m.context['solution'] = ''

@m.hook(fixme())
def solve(state):
    if False:
        return 10
    flag_byte = state.cpu.AL - fixme()
    m.context['solution'] += chr(flag_byte)
    fixme()
m.verbosity = 0
procs = 1
m.run(procs)
print(m.context['solution'])