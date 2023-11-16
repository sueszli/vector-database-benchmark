def readFile():
    if False:
        i = 10
        return i + 15
    try:
        with open('text.txt', encoding='utf-8') as f:
            read_lines = f.read().splitlines()
            read_exist = True
    except FileNotFoundError:
        read_lines = []
        read_exist = False
    return (read_exist, read_lines)

def printLines(list_text):
    if False:
        print('Hello World!')
    print('\n Texto introducido hasta ahora:')
    _ = [print(n) for n in list_text]
    print('-' * 25)
(exist, lines) = readFile()
if exist:
    clean = []
    while clean not in ['S', 'N']:
        clean = input('El archivo ya existe, ¿quieres conservar su contendio? [s/n]').upper()
        if clean == 'S':
            printLines(lines)
        else:
            lines = []
line = input('Introduce un texto:\n')
lines.append(line)
run = True
while run:
    answer = input('¿Quieres seguir introduciendo texto? [s/n]').upper()
    if answer == 'S':
        printLines(lines)
        line = input('Introduce un texto:\n')
        lines.append(line)
    elif answer == 'N':
        run = False
        printLines(lines)
        with open('text.txt', 'w', encoding='utf-8') as export:
            export.write('\n'.join(lines))
        print('Archivo guardado correctamente.')
    else:
        print('¡Error! Has introducido un valor no válido.')