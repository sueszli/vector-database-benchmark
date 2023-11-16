import string
my_letters_list = list(string.ascii_letters.lower())
new_characters = ['ñ']
my_letters_list.extend(new_characters)

def elimina_tildes_may_esp(word):
    if False:
        while True:
            i = 10
    word = word.replace('á', 'a')
    word = word.replace('é', 'e')
    word = word.replace('í', 'i')
    word = word.replace('ó', 'o')
    word = word.replace('ú', 'u')
    word = word.lower()
    word = word.replace(' ', '')
    return word

def get_heterograma(word):
    if False:
        print('Hello World!')
    heterograma = bool
    word = elimina_tildes_may_esp(word)
    for char in my_letters_list:
        if word.count(char) <= 1:
            heterograma = True
        elif word.count(char) > 1:
            heterograma = False
            break
    return heterograma

def get_isograma(word):
    if False:
        for i in range(10):
            print('nop')
    isograma = bool
    my_dict_char = {}
    word = elimina_tildes_may_esp(word)
    for char in word:
        if word.count(char) != 0:
            my_dict_char[char] = word.count(char)
            valor_maximo = max(my_dict_char.values())
            valor_minimo = min(my_dict_char.values())
            if valor_maximo == valor_minimo:
                isograma = True
            else:
                isograma = False
    return isograma

def get_pangrama(word):
    if False:
        for i in range(10):
            print('nop')
    pangrama = bool
    my_dict_char = {}
    word = elimina_tildes_may_esp(word)
    for char in my_letters_list:
        my_dict_char[char] = word.count(char)
    valor_minimo = min(my_dict_char.values())
    if valor_minimo != 0:
        pangrama = True
    else:
        pangrama = False
    return pangrama
palabra = input('Introduzca una palabra o frase para determinar si es heterograma, isograma y pangrama: ')
if get_heterograma(palabra) == False:
    print(f'"{palabra}" NO es un heterograma')
else:
    print(f'"{palabra}" SI es un heterograma')
if get_isograma(palabra) == False:
    print(f'"{palabra}" NO es un isograma')
else:
    print(f'"{palabra}" SI es un isograma')
if get_pangrama(palabra) == False:
    print(f'"{palabra}" NO es un pangrama')
else:
    print(f'"{palabra}" SI es un pangrama')