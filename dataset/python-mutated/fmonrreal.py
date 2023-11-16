def obtenerValoresURL(url: str):
    if False:
        for i in range(10):
            print('nop')
    variables = []
    try:
        variablesEnURL = url.split('?')[1]
        for v in variablesEnURL.split('&'):
            variables.append(v.split('=')[1])
        print(variables)
    except:
        print('Esa URL no tiene parámetros los cuáles desglosar')
obtenerValoresURL('https://retosdeprogramacion.com')
obtenerValoresURL('https://retosdeprogramacion.com?year=2023&challenge=0')