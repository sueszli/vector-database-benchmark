def escalera(escalones):
    if False:
        print('Hello World!')
    espacio = ''
    if escalones == 0:
        print('__')
    elif escalones > 0:
        for i in range(0, escalones):
            espacio = espacio + '  '
        print(espacio + '_')
        for i in range(0, escalones):
            espacio = espacio[:-2]
            print(espacio + '_|')
    elif escalones < 0:
        print('_')
        espacio = espacio + ' '
        for i in range(0, -escalones):
            print(espacio + '|_')
            espacio = espacio + '  '