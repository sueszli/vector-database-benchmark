try:
    a
except Exc as b:
    b
except Exc2 as c:
    b

def foo():
    if False:
        return 10
    try:
        a
    except Exc as b:
        b