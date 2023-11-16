piedra = '🗿'
papel = '📄'
tijeras = '✂️'
lagarto = '🦎'
spock = '🖖'
apuestas = [tijeras, papel, piedra, lagarto, spock]

def misma_paridad(j1: int, j2: int) -> bool:
    if False:
        return 10
    'Devuelve True si ambos parametros son pares, o impares.\n    Devuelve False si uno es par y el otro impar'
    return j1 % 2 == j2 % 2

def ganador_jugada(apuesta_player_1: str, apuesta_player_2: str) -> str:
    if False:
        print('Hello World!')
    'Dados un par de apuestas, determina el ganador'
    j1 = apuestas.index(apuesta_player_1)
    j2 = apuestas.index(apuesta_player_2)
    if j1 == j2:
        return 'Tie'
    if misma_paridad(j1, j2):
        return 'Player 2' if j1 < j2 else 'Player 1'
    else:
        return 'Player 1' if j1 < j2 else 'Player 2'

def ganador_serie(apuestas: list) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Dada una lista de apuestas, determina qué jugador es el ganador'
    victorias: int = 0
    for (ap1, ap2) in apuestas:
        gana = ganador_jugada(ap1, ap2)
        if gana == 'Player 1':
            victorias += 1
        elif gana == 'Player 2':
            victorias -= 1
        else:
            pass
    if victorias > 0:
        return 'Player 1'
    if victorias < 0:
        return 'Player 2'
    return 'Tie'
print(ganador_serie([('🗿', '✂️')]))
print(ganador_serie([('🗿', '✂️'), ('✂️', '🗿')]))
print(ganador_serie([('🗿', '✂️'), ('✂️', '🗿'), ('📄', '✂️')]))