import string
import secrets

def password(lenght: int, upperCase: bool, numbers: bool, simbols: bool):
    if False:
        i = 10
        return i + 15
    if lenght < 8:
        lenght = 8
    if lenght > 16:
        lenght = 16
    charactesSpecial = string.punctuation
    number = string.digits
    if upperCase:
        alphabet = string.ascii_uppercase
    else:
        alphabet = string.ascii_lowercase
    if numbers:
        alphabet += number
    else:
        alphabet = string.ascii_letters
    if simbols:
        alphabet += charactesSpecial
    else:
        alphabet = string.ascii_letters
    password = ''.join((secrets.choice(alphabet) for i in range(lenght)))
    return password
print(password(7, True, True, True))
print(password(7, True, True, False))
print(password(7, False, True, True))
print(password(7, True, False, True))