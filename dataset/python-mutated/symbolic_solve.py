from manticore import Manticore

def fixme():
    if False:
        while True:
            i = 10
    raise Exception('Fill in the blanks!')
m = Manticore('multiple-styles')

@m.hook(fixme())
def solve(state):
    if False:
        for i in range(10):
            print('nop')
    flag_base = state.cpu.RBP - fixme()
    solution = ''
    for i in range(fixme()):
        symbolic_character = state.cpu.read_int(flag_base + i, 8)
        concrete_character = fixme()
        solution += chr(concrete_character)
    print(solution)
    m.terminate()
m.verbosity = 0
procs = 1
m.run(procs)