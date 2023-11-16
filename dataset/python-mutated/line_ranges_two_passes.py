def restrict_to_this_line(arg1, arg2, arg3):
    if False:
        i = 10
        return i + 15
    print('This should not be formatted.')
    print('Note that in the second pass, the original line range 9-11 will cover these print lines.')

def restrict_to_this_line(arg1, arg2, arg3):
    if False:
        print('Hello World!')
    print('This should not be formatted.')
    print('Note that in the second pass, the original line range 9-11 will cover these print lines.')