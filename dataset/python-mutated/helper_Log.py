from datetime import datetime
import os

class Result:
    OK = 1
    FAILURE = 2
    UPDATE = 3
    ALREADY_DONE = 4
    NEW = 5
    SEE_BELOW = 6
    IGNORED = 7

class Log:
    log_file = 'creation_log.txt'
    current_cursor_pos = 0
    terminal_columns = 120

    class style:
        BOLD = '\x1b[01m'
        BLACK = '\x1b[30m'
        RED = '\x1b[31m'
        GREEN = '\x1b[32m'
        YELLOW = '\x1b[33m'
        BLUE = '\x1b[34m'
        MAGENTA = '\x1b[35m'
        CYAN = '\x1b[36m'
        WHITE = '\x1b[37m'
        UNDERLINE = '\x1b[4m'
        RESET = '\x1b[0m'

    def Init(options):
        if False:
            print('Hello World!')
        try:
            (columns, rows) = os.get_terminal_size(0)
            Log.terminal_columns = columns
        except:
            pass
        if os.path.isfile(Log.log_file):
            os.rename(Log.log_file, Log.log_file + '.old')
        if options.debug:
            print('DEBUG: Used command line options: {}'.format(str(options)))
        with open(Log.log_file, 'w') as f:
            f.write('Used call options: {}\n'.format(str(options)))

    def PrintMainStart(options):
        if False:
            for i in range(10):
                print('nop')
        print('┌{}┐'.format('─' * (Log.terminal_columns - 2)))
        text = 'Auto generation of addon interface code'
        print('│ {}{}{}{}{}│'.format(Log.style.BOLD, Log.style.CYAN, text, Log.style.RESET, ' ' * (Log.terminal_columns - len(text) - 3)))
        text = 'Used options:'
        print('│ {}{}{}{}{}{}│'.format(Log.style.BOLD, Log.style.WHITE, Log.style.UNDERLINE, text, Log.style.RESET, ' ' * (Log.terminal_columns - len(text) - 3)))
        Log.__printUsedBooleanValueLine('force', options.force)
        Log.__printUsedBooleanValueLine('debug', options.debug)
        Log.__printUsedBooleanValueLine('commit', options.commit)
        print('└{}┘'.format('─' * (Log.terminal_columns - 2)))

    def PrintGroupStart(text):
        if False:
            print('Hello World!')
        print('─' * Log.terminal_columns)
        print('{}{} ...{}{}'.format(Log.style.CYAN, text, ' ' * (Log.terminal_columns - len(text) - 4), Log.style.RESET))
        with open(Log.log_file, 'a') as f:
            f.write('{}...\n'.format(text))

    def PrintBegin(text):
        if False:
            i = 10
            return i + 15
        dt_string = datetime.utcnow().strftime('%d/%m/%Y %H:%M:%S')
        Log.current_cursor_pos = len(text) + len(dt_string) + 3
        print('[{}{}{}] {}{}{}{}'.format(Log.style.MAGENTA, dt_string, Log.style.RESET, Log.style.WHITE, Log.style.BOLD, text, Log.style.RESET), end='')
        with open(Log.log_file, 'a') as f:
            f.write('[{}] {}: '.format(dt_string, text))

    def PrintFollow(text):
        if False:
            return 10
        Log.current_cursor_pos += len(text)
        print(Log.style.CYAN + text + Log.style.RESET, end='')
        with open(Log.log_file, 'a') as f:
            f.write('{} '.format(text))

    def PrintResult(result_type, result_text=None):
        if False:
            while True:
                i = 10
        text = ''
        color = Log.style.WHITE
        if result_type == Result.OK:
            text = 'OK'
            color = Log.style.GREEN
        elif result_type == Result.NEW:
            text = 'Created new'
            color = Log.style.CYAN
        elif result_type == Result.FAILURE:
            text = 'Failed'
            color = Log.style.RED
        elif result_type == Result.UPDATE:
            text = 'Updated'
            color = Log.style.YELLOW
        elif result_type == Result.ALREADY_DONE:
            text = 'Present and up to date'
            color = Log.style.GREEN
        elif result_type == Result.SEE_BELOW:
            text = 'See below'
            color = Log.style.BLUE
        elif result_type == Result.IGNORED:
            text = 'Ignored'
            color = Log.style.YELLOW
        print('{}{}{}{}'.format(color, Log.style.BOLD, text.rjust(Log.terminal_columns - Log.current_cursor_pos), Log.style.RESET))
        f = open(Log.log_file, 'a')
        f.write('{}\n'.format(text))
        if result_text:
            print('Results of call before:{}\n'.format(result_text))
            f.write('Results of call before:{}\n'.format(result_text))
        f.close()

    def PrintFatal(error_text):
        if False:
            for i in range(10):
                print('nop')
        dt_string = datetime.utcnow().strftime('%d/%m/%Y %H:%M:%S')
        Log.current_cursor_pos = len(error_text) + len(dt_string) + 3
        print('[{}{}{}] {}{}FATAL: {}{}'.format(Log.style.YELLOW, dt_string, Log.style.RESET, Log.style.RED, Log.style.BOLD, Log.style.RESET, error_text))
        with open(Log.log_file, 'a') as f:
            f.write('[{}] {}\n'.format(dt_string, error_text))

    def __printUsedBooleanValueLine(name, value):
        if False:
            for i in range(10):
                print('nop')
        text = '--{} = {}{}'.format(name, Log.style.RED + 'yes' if value else Log.style.GREEN + 'no', Log.style.RESET)
        print('│ {}{}{}{}{}│'.format(Log.style.BOLD, Log.style.WHITE, text, Log.style.RESET, ' ' * (Log.terminal_columns - len(text) + 6)))