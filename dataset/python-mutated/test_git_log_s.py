import os
import json
import unittest
import jc.parsers.git_log_s
from jc.exceptions import ParseError
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log.out'), 'r', encoding='utf-8') as f:
        generic_git_log = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-short.out'), 'r', encoding='utf-8') as f:
        generic_git_log_short = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-short-stat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_short_stat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-short-shortstat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_short_shortstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-medium.out'), 'r', encoding='utf-8') as f:
        generic_git_log_medium = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-medium-stat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_medium_stat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-medium-shortstat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_medium_shortstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-full.out'), 'r', encoding='utf-8') as f:
        generic_git_log_full = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-full-stat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_full_stat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-full-shortstat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_full_shortstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-fuller.out'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-fuller-stat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_stat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-fuller-shortstat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_shortstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-oneline.out'), 'r', encoding='utf-8') as f:
        generic_git_log_oneline = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-oneline-stat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_oneline_stat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-oneline-shortstat.out'), 'r', encoding='utf-8') as f:
        generic_git_log_oneline_shortstat = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-hash-in-message-fix.out'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_hash_in_message_fix = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-is-hash-regex-fix.out'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_is_hash_regex_fix = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-blank-author-fix.out'), 'r', encoding='utf-8') as f:
        generic_git_log_blank_author_fix = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-short-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_short_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-short-stat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_short_stat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-short-shortstat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_short_shortstat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-medium-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_medium_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-medium-stat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_medium_stat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-medium-shortstat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_medium_shortstat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-full-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_full_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-full-stat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_full_stat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-full-shortstat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_full_shortstat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-fuller-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-fuller-stat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_stat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-fuller-shortstat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_shortstat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-oneline-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_oneline_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-oneline-stat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_oneline_stat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-oneline-shortstat-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_oneline_shortstat_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-streaming-ignore-exceptions.json'), 'r', encoding='utf-8') as f:
        generic_git_log_streaming_ignore_exceptions_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-hash-in-message-fix-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_hash_in_message_fix_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-is-hash-regex-fix-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_fuller_is_hash_regex_fix_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/git-log-blank-author-fix-streaming.json'), 'r', encoding='utf-8') as f:
        generic_git_log_blank_author_fix_streaming_json = json.loads(f.read())

    def test_git_log_s_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'git_log' with no data\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse([], quiet=True)), [])

    def test_git_log_s_unparsable(self):
        if False:
            return 10
        data = 'unparsable data'
        g = jc.parsers.git_log_s.parse(data.splitlines(), quiet=True)
        with self.assertRaises(ParseError):
            list(g)

    def test_git_log_s_ignore_exceptions_success(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git log' with -qq (ignore_exceptions) option\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log.splitlines(), quiet=True, ignore_exceptions=True)), self.generic_git_log_streaming_ignore_exceptions_json)

    def test_ping_s_ignore_exceptions_error(self):
        if False:
            return 10
        "\n        Test 'ping' with -qq (ignore_exceptions) option option and error\n        "
        data_in = 'not git log'
        expected = json.loads('[{"_jc_meta":{"success":false,"error":"ParseError: Not git_log_s data","line":"not git log"}}]')
        self.assertEqual(list(jc.parsers.git_log_s.parse(data_in.splitlines(), quiet=True, ignore_exceptions=True)), expected)

    def test_git_log_s(self):
        if False:
            while True:
                i = 10
        "\n        Test 'git_log'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log.splitlines(), quiet=True)), self.generic_git_log_streaming_json)

    def test_git_log_short_s(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git_log --format=short'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_short.splitlines(), quiet=True)), self.generic_git_log_short_streaming_json)

    def test_git_log_short_stat_s(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git_log --format=short --stat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_short_stat.splitlines(), quiet=True)), self.generic_git_log_short_stat_streaming_json)

    def test_git_log_short_shortstat_s(self):
        if False:
            while True:
                i = 10
        "\n        Test 'git_log --format=short --shortstat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_short_shortstat.splitlines(), quiet=True)), self.generic_git_log_short_shortstat_streaming_json)

    def test_git_log_medium_s(self):
        if False:
            print('Hello World!')
        "\n        Test 'git_log --format=medium'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_medium.splitlines(), quiet=True)), self.generic_git_log_medium_streaming_json)

    def test_git_log_medium_stat_s(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git_log --format=medium --stat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_medium_stat.splitlines(), quiet=True)), self.generic_git_log_medium_stat_streaming_json)

    def test_git_log_medium_shortstat_s(self):
        if False:
            while True:
                i = 10
        "\n        Test 'git_log --format=medium --shortstat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_medium_shortstat.splitlines(), quiet=True)), self.generic_git_log_medium_shortstat_streaming_json)

    def test_git_log_full_s(self):
        if False:
            print('Hello World!')
        "\n        Test 'git_log --format=full'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_full.splitlines(), quiet=True)), self.generic_git_log_full_streaming_json)

    def test_git_log_full_stat_s(self):
        if False:
            print('Hello World!')
        "\n        Test 'git_log --format=full --stat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_full_stat.splitlines(), quiet=True)), self.generic_git_log_full_stat_streaming_json)

    def test_git_log_full_shortstat_s(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'git_log --format=full --shortstat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_full_shortstat.splitlines(), quiet=True)), self.generic_git_log_full_shortstat_streaming_json)

    def test_git_log_fuller_s(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git_log --format=fuller'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_fuller.splitlines(), quiet=True)), self.generic_git_log_fuller_streaming_json)

    def test_git_log_fuller_stat_s(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git_log --format=fuller --stat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_fuller_stat.splitlines(), quiet=True)), self.generic_git_log_fuller_stat_streaming_json)

    def test_git_log_fuller_shortstat_s(self):
        if False:
            print('Hello World!')
        "\n        Test 'git_log --format=fuller --shortstat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_fuller_shortstat.splitlines(), quiet=True)), self.generic_git_log_fuller_shortstat_streaming_json)

    def test_git_log_oneline_s(self):
        if False:
            print('Hello World!')
        "\n        Test 'git_log --format=oneline'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_oneline.splitlines(), quiet=True)), self.generic_git_log_oneline_streaming_json)

    def test_git_log_oneline_stat_s(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'git_log --format=oneline --stat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_oneline_stat.splitlines(), quiet=True)), self.generic_git_log_oneline_stat_streaming_json)

    def test_git_log_oneline_shortstat_s(self):
        if False:
            return 10
        "\n        Test 'git_log --format=oneline --shortstat'\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_oneline_shortstat.splitlines(), quiet=True)), self.generic_git_log_oneline_shortstat_streaming_json)

    def test_git_log_fuller_hash_in_message_fix(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'git_log --format=fuller --stat' fix for when a commit message\n        contains a line that is only a commit hash value.\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_fuller_hash_in_message_fix.splitlines(), quiet=True)), self.generic_git_log_fuller_hash_in_message_fix_streaming_json)

    def test_git_log_fuller_is_hash_fix(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'git_log --format=fuller --stat' fix for when a commit message\n        contains a line that evaluated as true to an older _is_hash regex\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_fuller_is_hash_regex_fix.splitlines(), quiet=True)), self.generic_git_log_fuller_is_hash_regex_fix_streaming_json)

    def test_git_log_blank_author_fix(self):
        if False:
            return 10
        "\n        Test 'git_log' fix for when a commit author has a blank name,\n        empty email, or both\n        "
        self.assertEqual(list(jc.parsers.git_log_s.parse(self.generic_git_log_blank_author_fix.splitlines(), quiet=True)), self.generic_git_log_blank_author_fix_streaming_json)
if __name__ == '__main__':
    unittest.main()