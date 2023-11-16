import os
import operator

def scan_dir(dir_path, challenges={}, languages={}, total=0, challenge_name=None, path_name=None) -> tuple:
    if False:
        i = 10
        return i + 15
    for path in os.scandir(dir_path):
        if path.is_dir():
            if 'Reto #' in path.name and path.name not in challenges:
                challenges[path.name] = 0
                challenge_name = path.name
            elif 'Reto #' not in path.name and path.name not in languages:
                languages[path.name] = 0
            (_, _, total) = scan_dir(path.path, challenges, languages, total, challenge_name, path.name)
        elif path_name in languages:
            total += 1
            if challenge_name is not None:
                challenges[challenge_name] += 1
            languages[path_name] += 1
    return (challenges, languages, total)
dir_path = os.path.dirname(__file__)
(challenges, languajes, total) = scan_dir(dir_path)
challenges = dict(sorted(challenges.items(), key=operator.itemgetter(1), reverse=True))
languajes = dict(sorted(languajes.items(), key=operator.itemgetter(1), reverse=True))
print(f'\nESTADÍSTICAS RETOS DE PROGRAMACIÓN:\n> {len(languajes.keys())} LENGUAJES ({total} CORRECCIONES)\n')
for challenge in challenges:
    print(f'> {challenge.upper()} ({challenges[challenge]}): {round(challenges[challenge] / total * 100, 2)}%')
print()
for languaje in languajes:
    print(f'> {languaje.upper()} ({languajes[languaje]}): {round(languajes[languaje] / total * 100, 2)}%')