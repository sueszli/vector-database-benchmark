def new_scores(scores: [int, int], player_1_wins_point: bool) -> []:
    if False:
        while True:
            i = 10
    player_A_score = scores[0] if player_1_wins_point else scores[1]
    player_B_score = scores[1] if player_1_wins_point else scores[0]
    if player_A_score == 5 or player_B_score == 5:
        return scores
    if player_A_score == 3 and player_B_score < 3:
        player_A_score = 5
    elif player_A_score == 4:
        player_A_score = 5
    elif player_A_score == 3 and player_B_score == 4:
        player_B_score = 3
    else:
        player_A_score += 1
    return [player_A_score, player_B_score] if player_1_wins_point else [player_B_score, player_A_score]
score_enum = {0: 'Love', 1: 15, 2: 30, 3: 40, 4: 'Advantage', 5: 'Win'}
sequence = ['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1']
scores = [0, 0]
print('Love All')
for point in sequence:
    scores = new_scores(scores, point == 'P1')
    if scores[0] == 3 and scores[1] == 3:
        print('Deuce')
    elif scores[0] == scores[1]:
        print(score_enum[scores[0]], ' All')
    elif scores[0] == 4:
        print('Ventaja P1')
    elif scores[1] == 4:
        print('Ventaja P2')
    elif scores[0] == 5:
        print('Gana P1')
    elif scores[1] == 5:
        print('Gana P2')
    else:
        print(score_enum[scores[0]], ' - ', score_enum[scores[1]])