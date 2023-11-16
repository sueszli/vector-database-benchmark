def obtenerValoresURL(url: str):
    if False:
        i = 10
        return i + 15
    try:
        variablesAndValuesURL = url.split('?')[1]
        variablesAndValuesSeparated = variablesAndValuesURL.split('&')
        values = list()
        "\n        '' Creamos un bucle para recorrer todas las variables de las lista con sus valores y volvemos a cortar por el =\n        '' para quedarnos sólo con los valores, cogemos el primer índice. Ej: ['Hola=Brais'] -> .split('=')[1] = 'Brais'\n        "
        for variableAndValue in variablesAndValuesSeparated:
            value = variableAndValue.split('=')[1]
            values.append(value)
        print(values)
    except:
        print('Esa URL no tiene parámetros los cuáles desglosar')
obtenerValoresURL('https://www.google.com/')
obtenerValoresURL('https://retosdeprogramacion.com?year=2023&challenge=0')