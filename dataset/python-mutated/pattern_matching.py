def pattern_matching():
    if False:
        print('Hello World!')
    match status:
        case 1:
            return '1'
        case [single]:
            return 'single'
        case [action, obj]:
            return 'act on obj'
        case Point(x=0):
            return 'class pattern'
        case {'text': message}:
            return 'mapping'
        case {'text': message, 'format': _}:
            return 'mapping'
        case _:
            return 'fallback'