import binascii
from pywhat.helper import read_json

def get_magic_nums(file_loc):
    if False:
        print('Hello World!')
    with open(file_loc, 'rb') as myfile:
        header = myfile.read(24)
        header = str(binascii.hexlify(header))[2:-1]
    return check_magic_nums(header)

def check_magic_nums(text):
    if False:
        for i in range(10):
            print('nop')
    for i in read_json('file_signatures.json'):
        to_check = i['Hexadecimal File Signature']
        if text.lower().startswith(to_check.lower()):
            return i
    return None