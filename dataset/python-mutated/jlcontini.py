"""
# Reto #2: EL PARTIDO DE TENIS
#### Dificultad: Media | Publicación: 09/01/23 | Corrección: 16/01/23

## Enunciado

```
/*
 * Escribe un programa que muestre cómo transcurre un juego de tenis y quién lo ha ganado.
 * El programa recibirá una secuencia formada por "P1" (Player 1) o "P2" (Player 2), según quien
 * gane cada punto del juego.
 * 
 * - Las puntuaciones de un juego son "Love" (cero), 15, 30, 40, "Deuce" (empate), ventaja.
 * - Ante la secuencia [P1, P1, P2, P2, P1, P2, P1, P1], el programa mostraría lo siguiente:
 *   15 - Love
 *   30 - Love
 *   30 - 15
 *   30 - 30
 *   40 - 30
 *   Deuce
 *   Ventaja P1
 *   Ha ganado el P1
 * - Si quieres, puedes controlar errores en la entrada de datos.
 * - Consulta las reglas del juego si tienes dudas sobre el sistema de puntos.   
 */
```
#### Tienes toda la información extendida sobre los retos de programación semanales en **[retosdeprogramacion.com/semanales2023](https://retosdeprogramacion.com/semanales2023)**.

Sigue las **[instrucciones](../../README.md)**, consulta las correcciones y aporta la tuya propia utilizando el lenguaje de programación que quieras.

> Recuerda que cada semana se publica un nuevo ejercicio y se corrige el de la semana anterior en directo desde **[Twitch](https://twitch.tv/mouredev)**. Tienes el horario en la sección "eventos" del servidor de **[Discord](https://discord.gg/mouredev)**.
"""
point_winner_list: [str] = ['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1']
GAME_SCORE: tuple = ('Love', 15, 30, 40, 'Deuce', 'Advantage')
TITLE: str = 'TENIS GAME SIMULATION'
print(f"\n\n{'*' * 70}\n{TITLE.center(70, '-')}\n{'*' * 70}\n")

def tennis_game_scoring(point_winner_list: list[str]) -> str:
    if False:
        return 10
    game_winner = ''
    while game_winner not in ['P1', 'P2', 'No one']:
        p1 = 0
        p2 = 0
        for point_winner in point_winner_list:
            if point_winner == 'P1':
                p1 += 1
            if point_winner == 'P2':
                p2 += 1
            if p1 > 3 or p2 > 3 or (p1 == 3 and p2 == 3):
                result = int(p1 - p2)
                if result == 0:
                    print(f'{GAME_SCORE[4]}')
                if result == 1:
                    print(f'{GAME_SCORE[5]} P1')
                if result == -1:
                    print(f'{GAME_SCORE[5]} P2')
                if result >= 2:
                    game_winner = 'P1'
                if result <= -2:
                    game_winner = 'P2'
            else:
                print(f'{GAME_SCORE[p1]} - {GAME_SCORE[p2]}')
        if game_winner not in ['P1', 'P2']:
            game_winner = 'No one'
            print('Missing points to finish game')
    return game_winner

def display_game_winner(game_winner) -> str:
    if False:
        for i in range(10):
            print('nop')
    display = str(f"\n  {'*' * 25}\n  {(str(game_winner) + str(' wins the game')).center(25, '-')}\n  {'*' * 25}\n  ")
    return display

def main():
    if False:
        print('Hello World!')
    game_winner = tennis_game_scoring(point_winner_list)
    print(display_game_winner(game_winner))
if __name__ == '__main__':
    main()