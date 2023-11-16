import os
import io

def read_fixture_lines(filename):
    if False:
        print('Hello World!')
    'Read lines of text from file.\n\n    :param filename: string name\n    :return: list of strings\n\n    '
    lines = []
    for line in open(filename):
        lines.append(line.strip())
    return lines

def read_fixture_files():
    if False:
        i = 10
        return i + 15
    'Read all files inside fixture_data directory.'
    fixture_dict = {}
    current_dir = os.path.dirname(__file__)
    fixture_dir = os.path.join(current_dir, 'fixture_data/')
    for filename in os.listdir(fixture_dir):
        if filename not in ['.', '..']:
            fullname = os.path.join(fixture_dir, filename)
            fixture_dict[filename] = read_fixture_lines(fullname)
    return fixture_dict