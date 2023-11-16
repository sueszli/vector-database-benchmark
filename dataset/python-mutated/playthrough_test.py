"""Re-run playthroughs and check for differences."""
import os
import re
from absl import logging
from absl.testing import absltest
from open_spiel.python.algorithms import generate_playthrough
import pyspiel
_DATA_DIR = 'open_spiel/integration_tests/playthroughs/'
_OPTIONAL_GAMES = frozenset(['hanabi', 'universal_poker'])
_AVAILABLE_GAMES = set(pyspiel.registered_names())
_MISSING_GAMES = set(['nfg_game', 'efg_game', 'restricted_nash_response'])
_SHORTNAME = '^GameType\\.short_name = "(.*)"$'

def _is_optional_game(basename):
    if False:
        return 10
    'Returns (bool, game_name or None).\n\n  Args:\n    basename: The basename of the file. It is assumed it starts with the game\n      name.\n  '
    for game_name in _OPTIONAL_GAMES:
        if basename.startswith(game_name):
            return (True, game_name)
    return (False, None)

def _playthrough_match(filename, regex):
    if False:
        for i in range(10):
            print('nop')
    'Returns the specified value fromm the playthrough.'
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    return re.search(regex, data, re.MULTILINE)

class PlaythroughTest(absltest.TestCase):

    def run_test(self, path, basename):
        if False:
            while True:
                i = 10
        'Instantiated for each test case in main, below.'
        (is_optional, game_name) = _is_optional_game(basename)
        if is_optional:
            if game_name not in _AVAILABLE_GAMES:
                logging.info('Skipping %s because %s is not built in.', basename, game_name)
                return
        file_path = os.path.join(path, basename)
        (expected, actual) = generate_playthrough.replay(file_path)
        for (line_num, (expected_line, actual_line)) in enumerate(zip(expected.split('\n'), actual.split('\n'))):
            self.assertEqual(expected_line, actual_line, msg='Wrong line {} in {}'.format(line_num, basename))
        self.assertMultiLineEqual(expected, actual)

    def test_all_games_tested(self):
        if False:
            print('Hello World!')
        'Verify that every game is present in the playthroughs.'
        test_srcdir = os.environ.get('TEST_SRCDIR', '')
        path = os.path.join(test_srcdir, _DATA_DIR)
        basenames = set(os.listdir(path))
        missing_games = set(_AVAILABLE_GAMES) - set(_MISSING_GAMES) - set((_playthrough_match(os.path.join(path, basename), _SHORTNAME)[1] for basename in basenames))
        self.assertEmpty(missing_games, msg='These games do not have playthroughs.Create playthroughs using generate_new_playthrough.sh')

def _add_tests():
    if False:
        i = 10
        return i + 15
    'Adds a test for each playthrough to the test class (above).'
    test_srcdir = os.environ.get('TEST_SRCDIR', '')
    path = os.path.join(test_srcdir, _DATA_DIR)
    basenames = sorted(os.listdir(path))
    if len(basenames) < 40:
        raise ValueError(f'Playthroughs are missing from {path}')
    for basename in basenames:
        test_name = f'test_playthrough_{basename}'
        test_func = lambda self, basename=basename: self.run_test(path, basename)
        test_func.__name__ = test_name
        setattr(PlaythroughTest, test_name, test_func)
if __name__ == '__main__':
    _add_tests()
    absltest.main()