"""* Escribe un programa que sea capaz de generar contraseñas de forma aleatoria.
* Podrás configurar generar contraseñas con los siguientes parámetros:
* - Longitud: Entre 8 y 16.
* - Con o sin letras mayúsculas.
* - Con o sin números.
* - Con o sin símbolos.
* (Pudiendo combinar todos estos parámetros entre ellos)
"""
import random

def generador_password():
    if False:
        for i in range(10):
            print('nop')
    'Genera una contraseña de entre 8 y 16 caracteres, que pueden incluir letras\n  mayúsculas, minúsculas, números y símbolos según la decisión del usuario'
    cant = int(input('Ingresa cantidad de caracteres (8-16): \n'))
    while cant < 8 or cant > 16:
        print(f'Cantidad de caracteres no permitida, reintente.')
        cant = int(input('Ingresa cantidad de caracteres (8-16): \n'))
    letras_min = input('¿Desea que la contraseña incluya minúsculas (S/N): ')
    letras_may = input('¿Desea que la contraseña incluya mayúsculas (S/N): ')
    letras_num = input('¿Desea que la contraseña incluya números (S/N): ')
    letras_simb = input('¿Desea que la contraseña incluya símbolos (S/N): ')
    caracteres = []
    if letras_min.upper() == 'S':
        caracteres.append('min')
    if letras_may.upper() == 'S':
        caracteres.append('may')
    if letras_num.upper() == 'S':
        caracteres.append('num')
    if letras_simb.upper() == 'S':
        caracteres.append('simb')
    caracteres_random = random.choices(caracteres, k=cant)
    min = 'abcdefghijklmnopqrstuvwxyz'
    may = min.upper()
    num = '0123456789'
    simb = '|\'¿´+{,.-°!"#$%&/()=?¡¨*[];:_ }'
    for e in range(len(caracteres_random)):
        if caracteres_random[e] == 'min':
            caracteres_random[e] = str(random.choice(min))
        elif caracteres_random[e] == 'may':
            caracteres_random[e] = str(random.choice(may))
        elif caracteres_random[e] == 'num':
            caracteres_random[e] = str(random.choice(num))
        else:
            caracteres_random[e] = str(random.choice(simb))
    print(f"*** Contraseña generada*** \n {''.join(caracteres_random)}")
generador_password()