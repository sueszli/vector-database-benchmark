import random
list_points = ['Love', '15', '30', '40']

def tennisMatch(sequence):
    if False:
        print('Hello World!')
    print('¡Comienza el partido!')
    points_p1 = 0
    points_p2 = 0
    for p in sequence:
        if p == 'P1':
            points_p1 += 1
            if points_p1 == points_p2 and points_p1 >= 3:
                print('Deuce')
            elif points_p1 < 4 and points_p2 < 4:
                print(list_points[points_p1] + ' - ' + list_points[points_p2])
            elif points_p1 == 4 and points_p2 < 3 or (points_p1 > points_p2 + 1 and points_p2 >= 3):
                print('Ha ganado el P1')
                break
            elif points_p1 == points_p2 + 1:
                print('Ventaja P1')
            else:
                print('ERROR')
        elif p == 'P2':
            points_p2 += 1
            if points_p1 == points_p2 and points_p2 >= 3:
                print('Deuce')
            elif points_p2 < 4 and points_p1 < 4:
                print(list_points[points_p1] + ' - ' + list_points[points_p2])
            elif points_p2 == 4 and points_p1 < 3 or (points_p2 > points_p1 + 1 and points_p1 >= 3):
                print('Ha ganado el P2')
                break
            elif points_p2 == points_p1 + 1:
                print('Ventaja P2')
            else:
                print('ERROR')
        else:
            print('ERROR')
            break
tennisMatch(['P1', 'P2', 'P1', 'P1', 'P1'])
tennisMatch(['P2', 'P1', 'P1', 'P2', 'P1', 'P2', 'P2', 'P2'])
tennisMatch(['P1', 'P2', 'P2', 'P2', 'P1', 'P2'])
tennisMatch(['P1', 'P1', 'P1', 'P2', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1'])
tennisMatch(['P1', 'P1', 'P2', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P2', 'P2', 'P2'])