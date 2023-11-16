print('Hola, mundo!')
mi_texto = '¡Hola desde Python!'
mi_entero = 42
mi_decimal = 3.14
mi_booleano = True
MI_CONSTANTE = 10
if mi_entero > 50:
    print('El número es mayor que 50')
elif mi_entero < 50:
    print('El número es menor que 50')
else:
    print('El número es igual a 50')
mi_lista = [1, 2, 3, 4, 5]
mi_lista_texto = ['Manzana', 'Banana', 'Naranja']
mi_tupla = (1, 'Tupla')
mi_set = {'Rojo', 'Verde', 'Azul'}
mi_diccionario = {'clave1': 'valor1', 'clave2': 'valor2'}
for elemento in mi_lista:
    print(elemento)
for elemento in mi_lista_texto:
    print(elemento)
contador = 0
while contador < 3:
    print('Contador:', contador)
    contador += 1

def funcion_sin_parametros():
    if False:
        while True:
            i = 10
    print('Función sin parámetros')

def funcion_con_parametros(param1, param2):
    if False:
        i = 10
        return i + 15
    print('Parámetro 1:', param1)
    print('Parámetro 2:', param2)

def funcion_con_retorno(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a + b

class Persona:

    def __init__(self, nombre, edad):
        if False:
            while True:
                i = 10
        self.nombre = nombre
        self.edad = edad
try:
    resultado = mi_entero / 0
except Exception as e:
    print('Error:', e)