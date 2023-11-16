@(why := did[python].allow.this[111])
def function():
    if False:
        return 10
    return None

@(why := (lambda x: x))
def grr():
    if False:
        i = 10
        return i + 15
    return None

@iwonder
def grumble():
    if False:
        i = 10
        return i + 15
    return 'grrrrr'