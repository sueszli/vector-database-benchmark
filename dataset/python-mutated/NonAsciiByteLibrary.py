def in_exception():
    if False:
        return 10
    raise Exception(b'hyv\xe4')

def in_return_value():
    if False:
        print('Hello World!')
    return b'ty\xf6paikka'

def in_message():
    if False:
        i = 10
        return i + 15
    print(b'\xe4iti')

def in_multiline_message():
    if False:
        for i in range(10):
            print('nop')
    print(b'\xe4iti\nis\xe4')