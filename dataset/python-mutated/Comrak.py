import os
archivos_TXT = []

def escribir(nombre_txt, tipo):
    if False:
        i = 10
        return i + 15
    with open(nombre_txt, tipo) as f:
        print('escribe lo que quieras presiona enterpara saltar de linea y ME QUIERO IR para salir: ')
        while True:
            nueva_linea = input()
            if nueva_linea == 'ME QUIERO IR':
                break
            else:
                f.write(f'{nueva_linea} \n')
    f.close()

def leer(nombre_txt, tipo):
    if False:
        print('Hello World!')
    print('continua donde te quedaste \n')
    with open(nombre_txt, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            print(line.strip())
    f.close
    escribir(nombre_txt, tipo)
for archivos in os.listdir():
    if '.txt' in archivos:
        archivos_TXT.append(archivos)
if len(archivos_TXT) > 0:
    while True:
        abrir = input(f"Tenemos los siguientes TXT,{' '.join(map(str, archivos_TXT))} escribe el nombre de cual te gustaria abrir: ")
        try:
            leer(abrir, 'a')
            break
        except FileNotFoundError:
            print('no puedo encontrar ese archivo, y si lo intentamos de vuelta recuerda escribirlo con la extension incluida')
else:
    print(f'parece que no tienes ningun archivo .txt aca te voy a crear uno para que empiezes')
    escribir('text.txt', 'w')