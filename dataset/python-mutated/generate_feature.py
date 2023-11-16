from glob import glob
import json
from pathlib import Path
import sys
'Convert existing intent tests to behave tests.'
TEMPLATE = '\n  Scenario: {scenario}\n    Given an english speaking user\n     When the user says "{utterance}"\n     Then "{skill}" should reply with dialog from "{dialog_file}.dialog"\n'

def json_files(path):
    if False:
        i = 10
        return i + 15
    'Generator function returning paths of all json files in a folder.'
    for json_file in sorted(glob(str(Path(path, '*.json')))):
        yield Path(json_file)

def generate_feature(skill, skill_path):
    if False:
        while True:
            i = 10
    'Generate a feature file provided a skill name and a path to the skill.\n    '
    test_path = Path(skill_path, 'test', 'intent')
    case = []
    if test_path.exists() and test_path.is_dir():
        for json_file in json_files(test_path):
            with open(str(json_file)) as test_file:
                test = json.load(test_file)
                if 'utterance' and 'expected_dialog' in test:
                    utt = test['utterance']
                    dialog = test['expected_dialog']
                    if isinstance(dialog, list):
                        dialog = dialog[0]
                    case.append((json_file.name, utt, dialog))
    output = ''
    if case:
        output += 'Feature: {}\n'.format(skill)
    for c in case:
        output += TEMPLATE.format(skill=skill, scenario=c[0], utterance=c[1], dialog_file=c[2])
    return output
if __name__ == '__main__':
    print(generate_feature(*sys.argv[1:]))