"""The Caesar Shift Cipher example Fire CLI.

This module demonstrates the use of Fire without specifying a target component.
Notice how the call to Fire() in the main method doesn't indicate a component.
So, all local and global variables (including all functions defined in the
module) are made available as part of the Fire CLI.

Example usage:
cipher rot13 'Hello world!'  # Uryyb jbeyq!
cipher rot13 'Uryyb jbeyq!'  # Hello world!
cipher caesar-encode 1 'Hello world!'  # Ifmmp xpsme!
cipher caesar-decode 1 'Ifmmp xpsme!'  # Hello world!
"""
import fire

def caesar_encode(n=0, text=''):
    if False:
        i = 10
        return i + 15
    return ''.join((_caesar_shift_char(n, char) for char in text))

def caesar_decode(n=0, text=''):
    if False:
        print('Hello World!')
    return caesar_encode(-n, text)

def rot13(text):
    if False:
        for i in range(10):
            print('nop')
    return caesar_encode(13, text)

def _caesar_shift_char(n=0, char=' '):
    if False:
        while True:
            i = 10
    if not char.isalpha():
        return char
    if char.isupper():
        return chr((ord(char) - ord('A') + n) % 26 + ord('A'))
    return chr((ord(char) - ord('a') + n) % 26 + ord('a'))

def main():
    if False:
        print('Hello World!')
    fire.Fire(name='cipher')
if __name__ == '__main__':
    main()