if not (not a):  # SIM208
    pass

if not (not (a == b)):  # SIM208
    pass

if not a:  # OK
    pass

if not a == b:  # OK
    pass

if not a != b:  # OK
    pass

a = not not b  # SIM208

f(not not a)  # SIM208

if 1 + (not (not a)):  # SIM208
    pass
