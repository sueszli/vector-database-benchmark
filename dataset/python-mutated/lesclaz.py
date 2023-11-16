__score = ('Love', 15, 30, 40, 'Deuce', 'Ventaja')

def main(sequence: list[str]) -> None:
    if False:
        while True:
            i = 10
    _player_1 = 0
    _player_2 = 0
    if not 'P1' in sequence or not 'P2' in sequence:
        raise ValueError('Debe proporcionar una secuencia de jugadasganadas por un jugador en concreto ex: ["P1", "P1", "P1", "P2", "P1", ...]')
    for _ in sequence:
        if _ == 'P1':
            _player_1 += 1
        elif _ == 'P2':
            _player_2 += 1
        _advantage = _player_1 - _player_2 if _player_1 - _player_2 >= 0 else (_player_1 - _player_2) * -1
        if (_player_1 >= 3 and _player_2 >= 3) and _player_1 == _player_2:
            print('Deuce')
        elif _player_1 <= 3 and _player_2 <= 3:
            print(f'{__score[_player_1]} - {__score[_player_2]}')
        elif (_player_1 >= 2 or _player_2 >= 2) and _advantage == 2:
            print('Ha ganado el', 'P1' if _player_1 > _player_2 else 'P2')
            break
        elif (_player_1 >= 3 or _player_2 >= 3) and _advantage == 1:
            print('Ventaja', 'P1' if _player_1 > _player_2 else 'P2')
if __name__ == '__main__':
    main(['P1', 'P1', 'P2', 'P2', 'P1', 'P2', 'P1', 'P1'])