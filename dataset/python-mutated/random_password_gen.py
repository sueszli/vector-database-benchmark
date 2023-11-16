import random
import math
alpha = 'abcdefghijklmnopqrstuvwxyz'
num = '0123456789'
special = '@#$%&*'
pass_len = int(input('Enter Password Length'))
alpha_len = pass_len // 2
num_len = math.ceil(pass_len * 30 / 100)
special_len = pass_len - (alpha_len + num_len)
password = []

def generate_pass(length, array, is_alpha=False):
    if False:
        for i in range(10):
            print('nop')
    for i in range(length):
        index = random.randint(0, len(array) - 1)
        character = array[index]
        if is_alpha:
            case = random.randint(0, 1)
            if case == 1:
                character = character.upper()
        password.append(character)
generate_pass(alpha_len, alpha, True)
generate_pass(num_len, num)
generate_pass(special_len, special)
random.shuffle(password)
gen_password = ''
for i in password:
    gen_password = gen_password + str(i)
print(gen_password)