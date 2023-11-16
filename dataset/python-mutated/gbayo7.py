from enum import Enum

class player(Enum):
    P1 = (1,)
    P2 = 2

def tennisGame(plays: list):
    if False:
        print('Hello World!')
    output = ''
    last_P1 = 0
    last_P2 = 0
    finished = False
    win_msg = 'Ha ganado '
    for (index, winner) in enumerate(plays):
        if winner == player.P1:
            if last_P1 == 0:
                last_P1 = 15
            elif last_P1 == 15:
                last_P1 = 30
            elif last_P1 == 30:
                last_P1 = 40
            elif last_P1 == 40:
                if last_P2 != 40 and last_P2 != 50:
                    finished = True
                    win_msg += 'P1'
                elif last_P2 == 50:
                    last_P1 = 40
                    last_P2 = 40
                elif last_P2 == 40:
                    last_P1 = 50
            elif last_P1 == 50:
                finished = True
                win_msg += 'P1'
        elif winner == player.P2:
            if last_P2 == 0:
                last_P2 = 15
            elif last_P2 == 15:
                last_P2 = 30
            elif last_P2 == 30:
                last_P2 = 40
            elif last_P2 == 40:
                if last_P1 != 40 and last_P1 != 50:
                    finished = True
                    win_msg += 'P2'
                elif last_P1 == 50:
                    last_P1 = 40
                    last_P2 = 40
                elif last_P1 == 40:
                    last_P2 = 50
                elif last_P2 == 50:
                    finished = True
                    win_msg += 'P2'
        if finished:
            output += win_msg
            break
        if last_P1 == 0:
            p_p1 = 'Love'
        elif last_P1 == 50:
            p_p1 = 'Ad P1'
        else:
            p_p1 = str(last_P1)
        if last_P2 == 0:
            p_p2 = 'Love'
        elif last_P2 == 50:
            p_p2 = 'Ad P1'
        else:
            p_p2 = str(last_P2)
        if p_p1 == p_p2:
            output += 'Deuce\n'
        elif 'Ad' in p_p1:
            output += p_p1 + '\n'
        elif 'Ad' in p_p2:
            output += p_p2 + '\n'
        else:
            output += f'{p_p1} - {p_p2}\n'
    print(output)

def main():
    if False:
        print('Hello World!')
    secuence = [player.P1, player.P1, player.P2, player.P2, player.P1, player.P2, player.P1, player.P1]
    tennisGame(secuence)
main()