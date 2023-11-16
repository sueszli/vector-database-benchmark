"""
Este es el grafo que define las reglas de quien vence a quien en una jugada.
Dentro del diccionario la llave define el objeto y el valor define el conjunto de 
objetos a los que el objeto llave puede eliminar.
"""
rule_graph = {'✂️': ['📄', '🦎'], '📄': ['🗿', '🖖'], '🗿': ['🦎', '✂️'], '🦎': ['🖖', '📄'], '🖖': ['✂️', '🗿']}

def play(p1, p2):
    if False:
        while True:
            i = 10
    '\n    Se encarga de validar cada jagada del juego\n    PIEDRA, PAPEL, TIJERA, LAGARTO, SPOCK.\n\n    Args:\n        p1: (str) Objeto que utilizó el jugador 1 (✂️, 📄, 🗿, 🦎 ó 🖖)\n        p2: (str) Objeto que utilizó el jugador 1 (✂️, 📄, 🗿, 🦎 ó 🖖)\n\n    Returns:\n        result: (str) Objeto que ganó la jugada definido por las reglas en el grafo de reglas\n        o si hubo un empate en dicha jugada\n    '
    if p1 == p2:
        return 0
    for value in rule_graph[p1]:
        if p2 == value:
            return p1
    return p2

def play_game(plays: list=[]):
    if False:
        while True:
            i = 10
    "\n    Define que jugador gana una partida del juego\n    PIEDRA, PAPEL, TIJERA, LAGARTO, SPOCK.\n\n    Para identificar a cada objeto usted debe emplear la siguiente nomenclatura:\n    ✂️ - para tijera,\n    📄 - para papel,\n    🗿 - para piedra,\n    🦎 - para lagarto,\n    🖖 - para spock,\n\n    Ej. [('🗿', '✂️'), ('✂️', '🗿'), ('📄', '🖖')]\n\n    Args:\n        plays: (list) Es una lista de tuplas que corresponden a cada jugada \n\n    Returns:\n        result: (str) El resultado del juego\n    "
    points_p1 = 0
    points_p2 = 0
    for (p1, p2) in plays:
        result = play(p1, p2)
        if result == p1:
            points_p1 += 1
        elif result == p2:
            points_p2 += 1
    if points_p1 > points_p2:
        return 'Player 1'
    elif points_p2 > points_p1:
        return 'Player 2'
    return 'Tie'
result = play_game([('🗿', '✂️'), ('✂️', '🗿'), ('📄', '✂️')])
print(result)