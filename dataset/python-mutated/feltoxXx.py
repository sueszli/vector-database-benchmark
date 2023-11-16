def game_resolver(secuencia: list):
    if False:
        while True:
            i = 10
    p1_puntaje = 0
    p2_puntaje = 0
    finished = False
    error = False
    puntajes = ['Love', '15', '30', '40']
    for punto in secuencia:
        if punto == 'P1':
            p1_puntaje += 1
        else:
            p2_puntaje += 1
        if p1_puntaje >= 3 and p2_puntaje >= 3:
            if not finished and abs(p1_puntaje - p2_puntaje) <= 1:
                print('Deuce' if p1_puntaje == p2_puntaje else 'Ventaja P1' if p1_puntaje > p2_puntaje else 'Ventaja P2')
            else:
                finished = True
        elif p1_puntaje < 4 and p2_puntaje < 4:
            print(f'{puntajes[p1_puntaje]} - {puntajes[p2_puntaje]}')
        else:
            finished = True
    if error or not finished:
        print('Los puntos no son correctos o faltan puntos para terminar el partido')
    else:
        print('Ha ganado el P1' if p1_puntaje > p2_puntaje else 'Ha ganado el P2')

def main():
    if False:
        for i in range(10):
            print('nop')
    secuencia = ['P1', 'P1', 'P2', 'P2', 'P2', 'P1', 'P1', 'P2', 'P1', 'P1']
    game_resolver(secuencia)
if __name__ == '__main__':
    main()