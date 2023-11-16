def obtener_valores(url):
    if False:
        i = 10
        return i + 15
    return [p.split('=')[1] for p in url.split('?')[1].split('&')]
print(obtener_valores('https://retosdeprogramacion.com?year=2023&challenge=0'))