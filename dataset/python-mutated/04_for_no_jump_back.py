def instruction_sequence_value(instrs, a, b):
    if False:
        while True:
            i = 10
    for instr in instrs:
        if a:
            a = 6
        elif b:
            return 0
        pass
    return a
assert instruction_sequence_value([], True, True) == 1
assert instruction_sequence_value([1], True, True) == 6
assert instruction_sequence_value([1], False, True) == 0
assert instruction_sequence_value([1], False, False) == False