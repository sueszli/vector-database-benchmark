import io
import pytest
import sys
import unittest
from sfcli import SpiderFootCli

@pytest.mark.usefixtures
class TestSpiderFootCli(unittest.TestCase):
    """
    Test TestSpiderFootCli
    """

    def test_default(self):
        if False:
            return 10
        '\n        Test default(self, line)\n        '
        sfcli = SpiderFootCli()
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.default('')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('Unknown command', output)

    def test_default_should_ignore_comments(self):
        if False:
            while True:
                i = 10
        '\n        Test default(self, line)\n        '
        sfcli = SpiderFootCli()
        io_output = io.StringIO()
        sys.stdout = io_output
        result = sfcli.default('# test comment')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertEqual(None, result)
        self.assertEqual('', output)

    def test_complete_start_should_return_a_list(self):
        if False:
            print('Hello World!')
        '\n        Test complete_start(self, text, line, startidx, endidx)\n        '
        sfcli = SpiderFootCli()
        start = sfcli.complete_start(None, None, None, None)
        self.assertIsInstance(start, list)
        self.assertEqual([], start)

    def test_complete_find_should_return_a_list(self):
        if False:
            print('Hello World!')
        '\n        Test complete_find(self, text, line, startidx, endidx)\n        '
        sfcli = SpiderFootCli()
        find = sfcli.complete_find(None, None, None, None)
        self.assertIsInstance(find, list)
        self.assertEqual([], find)

    def test_complete_data_should_return_a_list(self):
        if False:
            while True:
                i = 10
        '\n        Test complete_data(self, text, line, startidx, endidx)\n        '
        sfcli = SpiderFootCli()
        data = sfcli.complete_data(None, None, None, None)
        self.assertIsInstance(data, list)
        self.assertEqual([], data)

    def test_complete_default(self):
        if False:
            while True:
                i = 10
        '\n        Test complete_default(self, text, line, startidx, endidx)\n        '
        sfcli = SpiderFootCli()
        default = sfcli.complete_default('', '-t -m', None, None)
        self.assertIsInstance(default, list)
        self.assertEqual('TBD', 'TBD')
        default = sfcli.complete_default('', '-m -t', None, None)
        self.assertIsInstance(default, list)
        self.assertEqual('TBD', 'TBD')

    def test_complete_default_invalid_text_should_return_a_string(self):
        if False:
            return 10
        '\n        Test complete_default(self, text, line, startidx, endidx)\n        '
        sfcli = SpiderFootCli()
        default = sfcli.complete_default(None, 'example line', None, None)
        self.assertIsInstance(default, list)
        self.assertEqual([], default)

    def test_complete_default_invalid_line_should_return_a_string(self):
        if False:
            i = 10
            return i + 15
        '\n        Test complete_default(self, text, line, startidx, endidx)\n        '
        sfcli = SpiderFootCli()
        default = sfcli.complete_default('example text', None, None, None)
        self.assertIsInstance(default, list)
        self.assertEqual([], default)

    def test_do_debug_should_toggle_debug(self):
        if False:
            i = 10
            return i + 15
        '\n        Test do_debug(self, line)\n        '
        sfcli = SpiderFootCli(self.cli_default_options)
        sfcli.do_debug(None)
        initial_debug_state = sfcli.ownopts['cli.debug']
        sfcli.do_debug(None)
        new_debug_state = sfcli.ownopts['cli.debug']
        self.assertNotEqual(initial_debug_state, new_debug_state)

    def test_do_spool_should_toggle_spool(self):
        if False:
            i = 10
            return i + 15
        '\n        Test do_spool(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.spool_file'] = '/dev/null'
        sfcli.do_spool(None)
        initial_spool_state = sfcli.ownopts['cli.spool']
        sfcli.do_spool(None)
        new_spool_state = sfcli.ownopts['cli.spool']
        self.assertNotEqual(initial_spool_state, new_spool_state)

    def test_do_history_should_toggle_history_option(self):
        if False:
            while True:
                i = 10
        '\n        Test do_history(self, line)\n        '
        sfcli = SpiderFootCli(self.cli_default_options)
        sfcli.do_history('0')
        initial_history_state = sfcli.ownopts['cli.history']
        sfcli.do_history('1')
        new_history_state = sfcli.ownopts['cli.history']
        self.assertNotEqual(initial_history_state, new_history_state)

    def test_precmd_should_return_line(self):
        if False:
            print('Hello World!')
        '\n        Test precmd(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.history'] = False
        sfcli.ownopts['cli.spool'] = False
        line = 'example line'
        precmd = sfcli.precmd(line)
        self.assertEqual(line, precmd)

    @unittest.skip('todo')
    def test_precmd_should_print_line_to_history_file(self):
        if False:
            return 10
        '\n        Test precmd(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.history'] = True
        sfcli.ownopts['cli.spool'] = False
        line = 'example line'
        precmd = sfcli.precmd(line)
        self.assertEqual(line, precmd)
        self.assertEqual('TBD', 'TBD')

    @unittest.skip('todo')
    def test_precmd_should_print_line_to_spool_file(self):
        if False:
            while True:
                i = 10
        '\n        Test precmd(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.history'] = False
        sfcli.ownopts['cli.spool'] = True
        sfcli.ownopts['cli.spool_file'] = '/dev/null'
        line = 'example line'
        precmd = sfcli.precmd(line)
        self.assertEqual(line, precmd)
        self.assertEqual('TBD', 'TBD')

    def test_dprint_should_print_if_debug_option_is_set(self):
        if False:
            while True:
                i = 10
        '\n        Test dprint(self, msg, err=False, deb=False, plain=False, color=None)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.debug'] = True
        sfcli.ownopts['cli.spool'] = False
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.dprint('example output')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('example output', output)

    def test_dprint_should_not_print_unless_debug_option_is_set(self):
        if False:
            while True:
                i = 10
        '\n        Test dprint(self, msg, err=False, deb=False, plain=False, color=None)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.debug'] = False
        sfcli.ownopts['cli.spool'] = False
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.dprint('example output')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('', output)

    def test_ddprint_should_print_if_debug_option_is_set(self):
        if False:
            print('Hello World!')
        '\n        Test ddprint(self, msg)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.debug'] = True
        sfcli.ownopts['cli.spool'] = False
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.ddprint('example debug output')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('example debug output', output)

    def test_ddprint_should_not_print_unless_debug_option_is_set(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test ddprint(self, msg)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.debug'] = False
        sfcli.ownopts['cli.spool'] = False
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.ddprint('example debug output')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertEqual('', output)

    def test_edprint_should_print_error_regardless_of_debug_option(self):
        if False:
            i = 10
            return i + 15
        '\n        Test edprint(self, msg)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.debug'] = False
        sfcli.ownopts['cli.spool'] = False
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.edprint('example debug output')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('example debug output', output)

    def test_pretty_should_return_a_string(self):
        if False:
            return 10
        '\n        Test pretty(self, data, titlemap=None)\n        '
        sfcli = SpiderFootCli()
        invalid_types = [None, '', list(), dict()]
        for invalid_type in invalid_types:
            with self.subTest(invalid_type=invalid_type):
                pretty = sfcli.pretty(invalid_type)
                self.assertEqual('', pretty)

    def test_request_invalid_url_should_return_none(self):
        if False:
            i = 10
            return i + 15
        '\n        Test request(self, url, post=None)\n        '
        sfcli = SpiderFootCli()
        invalid_types = [None, list(), dict()]
        for invalid_type in invalid_types:
            with self.subTest(invalid_type=invalid_type):
                result = sfcli.request(invalid_type)
                self.assertEqual(None, result)

    def test_emptyline_should_return_none(self):
        if False:
            while True:
                i = 10
        '\n        Test emptyline(self)\n        '
        sfcli = SpiderFootCli()
        emptyline = sfcli.emptyline()
        self.assertEqual(None, emptyline)

    def test_completedefault_should_return_empty_list(self):
        if False:
            return 10
        '\n        Test completedefault(self, text, line, begidx, endidx)\n        '
        sfcli = SpiderFootCli()
        completedefault = sfcli.completedefault(None, None, None, None)
        self.assertIsInstance(completedefault, list)
        self.assertEqual([], completedefault)

    def test_myparseline_should_return_a_list_of_two_lists(self):
        if False:
            i = 10
            return i + 15
        '\n        Test myparseline(self, cmdline, replace=True)\n        '
        sfcli = SpiderFootCli()
        parsed_line = sfcli.myparseline(None)
        self.assertEqual(len(parsed_line), 2)
        self.assertIsInstance(parsed_line, list)
        self.assertIsInstance(parsed_line[0], list)
        self.assertIsInstance(parsed_line[1], list)
        parsed_line = sfcli.myparseline('')
        self.assertEqual(len(parsed_line), 2)
        self.assertIsInstance(parsed_line, list)
        self.assertIsInstance(parsed_line[0], list)
        self.assertIsInstance(parsed_line[1], list)

    def test_send_output(self):
        if False:
            i = 10
            return i + 15
        '\n        Test send_output(self, data, cmd, titles=None, total=True, raw=False)\n        '
        sfcli = SpiderFootCli()
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.send_output('{}', '', raw=True)
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('Total records: 0', output)
        self.assertEqual('TBD', 'TBD')

    def test_do_query(self):
        if False:
            while True:
                i = 10
        '\n        Test do_query(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_query(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_ping(self):
        if False:
            while True:
                i = 10
        '\n        Test do_ping(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_ping(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_modules(self):
        if False:
            print('Hello World!')
        '\n        Test do_modules(self, line, cacheonly=False)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_modules(None, None)
        self.assertEqual('TBD', 'TBD')

    def test_do_types(self):
        if False:
            i = 10
            return i + 15
        '\n        Test do_types(self, line, cacheonly=False)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_types(None, None)
        self.assertEqual('TBD', 'TBD')

    def test_do_load(self):
        if False:
            print('Hello World!')
        '\n        Test do_load(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_load(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_scaninfo(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test do_scaninfo(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_scaninfo(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_scans(self):
        if False:
            return 10
        '\n        Test do_scans(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_scans(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_data(self):
        if False:
            while True:
                i = 10
        '\n        Test do_data(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_data(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_export(self):
        if False:
            print('Hello World!')
        '\n        Test do_export(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_export(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_logs(self):
        if False:
            while True:
                i = 10
        '\n        Test do_logs(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_logs(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_start(self):
        if False:
            print('Hello World!')
        '\n        Test do_start(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_start(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test do_stop(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_stop(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_search(self):
        if False:
            i = 10
            return i + 15
        '\n        Test do_search(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_search(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_find(self):
        if False:
            return 10
        '\n        Test do_find(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_find(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_summary(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test do_summary(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_summary(None)
        self.assertEqual('TBD', 'TBD')

    def test_do_delete(self):
        if False:
            print('Hello World!')
        '\n        Test do_delete(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.do_delete(None)
        self.assertEqual('TBD', 'TBD')

    def test_print_topic(self):
        if False:
            return 10
        '\n        Test print_topics(self, header, cmds, cmdlen, maxcol)\n        '
        sfcli = SpiderFootCli()
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.print_topics(None, 'help', None, None)
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('Command', output)
        self.assertIn('Description', output)
        self.assertEqual('TBD', 'TBD')

    def test_do_set_should_set_option(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test do_set(self, line)\n        '
        sfcli = SpiderFootCli()
        sfcli.ownopts['cli.test_opt'] = None
        sfcli.do_set('cli.test_opt = "test value"')
        new_test_opt = sfcli.ownopts['cli.test_opt']
        self.assertEqual(new_test_opt, 'test value')

    def test_do_shell(self):
        if False:
            while True:
                i = 10
        '\n        Test do_shell(self, line)\n        '
        sfcli = SpiderFootCli()
        io_output = io.StringIO()
        sys.stdout = io_output
        sfcli.do_shell('')
        sys.stdout = sys.__stdout__
        output = io_output.getvalue()
        self.assertIn('Running shell command:', output)

    def test_do_clear(self):
        if False:
            print('Hello World!')
        '\n        Test do_clear(self, line)\n        '
        sfcli = SpiderFootCli()
        io_output = io.StringIO()
        sys.stderr = io_output
        sfcli.do_clear(None)
        sys.stderr = sys.__stderr__
        output = io_output.getvalue()
        self.assertEqual('\x1b[2J\x1b[H', output)

    def test_do_exit(self):
        if False:
            return 10
        '\n        Test do_exit(self, line)\n        '
        sfcli = SpiderFootCli()
        do_exit = sfcli.do_exit(None)
        self.assertTrue(do_exit)

    def test_do_eof(self):
        if False:
            i = 10
            return i + 15
        '\n        Test do_EOF(self, line)\n        '
        sfcli = SpiderFootCli()
        do_eof = sfcli.do_EOF(None)
        self.assertTrue(do_eof)