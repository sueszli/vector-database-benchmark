import time
games = [['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P2', 'P1', 'P2', 'P2', 'P2']]
"\ndef tennis_game(scores_game):\n    scores = ['Love', '15', '30', '40']\n    p1_score, p2_score = 0, 0\n    p1, p2 = 0, 0\n\n    for score in scores_game:\n        if score == 'P1':\n            p1_score += 1\n            print('Point for P1')\n        elif score == 'P2':\n            p2_score += 1\n            print('Point for P2')\n        else:\n            print(f'This input {score} is not valid')\n            return \n\n        if p1_score < 4:\n            p1 = scores[p1_score]\n        else:\n            p1 = p1_score\n        \n        if p2_score < 4:\n            p2 = scores[p2_score]\n        else:\n            p2 = p2_score\n\n        if (p1_score >= 4 or p2_score >= 4) and abs(p1_score - p2_score) >=2:\n            if p1_score > p2_score:\n                result_game = 'The winner is: Player 1'\n            else:\n                result_game = 'The winner is: Player 2'\n        \n            # refactor condicional anterior\n            # result_game = 'The winner is: Player 1' if p1_score > p2_score else 'The winner is: Player 2'\n        \n        elif (p1_score >= 3 or p2_score >= 3) and p1_score == p2_score:\n            result_game = 'Deuce'\n        elif (p1_score >= 3 or p2_score >= 3) and p1_score > p2_score:\n            result_game = 'Advantage P1'\n        elif (p1_score >= 3 or p2_score >= 3) and p1_score < p2_score:\n            result_game = 'Advantage P2'\n        else:            \n            result_game = (p1, p2)\n        \n        print(str(result_game) + '\n')\n        time.sleep(1)\n    return print('Full time!!')\n"

def check_data(game):
    if False:
        i = 10
        return i + 15
    if type(game) != list:
        return (False, 'It only accepts list() as an argument!')
    if not game:
        return (False, 'It only accepts not-empty list() as an argument!')
    if len(game) <= 4:
        return (False, 'The number of games played is not valid!')
    for i in game:
        if i != 'P1' and i != 'P2':
            return (False, 'The list of player is not valid!\nIt is only valid P1 and P2 for the points of Player1 and Player2 respectively.')
    return (True, '')

def tennis_game(game):
    if False:
        return 10
    (is_valid, error_msg) = check_data(game)
    if not is_valid:
        print(error_msg)
        print('Game Over!!\n')
    else:
        print('Star the match!!\n')
        scores = ['Love', '15', '30', '40']
        (p1_score, p2_score) = (0, 0)
        finished = False
        for point in game:
            time.sleep(1)
            if finished:
                print('The game is finished!\nNo more game points can be counted!\n')
                break
            if point == 'P1':
                p1_score += 1
                print('Point for P1')
            elif point == 'P2':
                p2_score += 1
                print('Point for P2')
            else:
                return print(f'This input {point} is not valid\n')
            if (p1_score >= 3 or p2_score >= 3) and p1_score == p2_score:
                result_game = 'Deuce\n'
            elif p1_score >= 4 or p2_score >= 4:
                if abs(p1_score - p2_score) >= 2:
                    result_game = 'The winner is: Player 1\n' if p1_score > p2_score else 'The winner is: Player 2\n'
                    finished = True
                elif p1_score > p2_score:
                    result_game = 'Advantage P1\n'
                else:
                    result_game = 'Advantage P2\n'
            else:
                result_game = f'({scores[p1_score]} - {scores[p2_score]})\n'
            print(result_game)
    return print('Full time!!\n')
for (index, game) in enumerate(games):
    time.sleep(1)
    print(index)
    tennis_game(game)