import re
import subprocess
import sys
verbosity = 0
suggestions = 1
ignore_prefixes = []

def verbose(*args):
    if False:
        while True:
            i = 10
    if verbosity:
        print(*args)

def very_verbose(*args):
    if False:
        return 10
    if verbosity > 1:
        print(*args)

class ErrorCollection:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.has_errors = False
        self.has_warnings = False
        self.prefix = ''

    def error(self, text):
        if False:
            i = 10
            return i + 15
        print('error: {}{}'.format(self.prefix, text))
        self.has_errors = True

    def warning(self, text):
        if False:
            print('Hello World!')
        print('warning: {}{}'.format(self.prefix, text))
        self.has_warnings = True

def git_log(pretty_format, *args):
    if False:
        i = 10
        return i + 15
    args = ['git', 'log'] + [arg for arg in args if '--pretty' not in args]
    args.append('--pretty=format:' + pretty_format)
    very_verbose('git_log', *args)
    for line in subprocess.Popen(args, stdout=subprocess.PIPE).stdout:
        yield line.decode().rstrip('\r\n')

def diagnose_subject_line(subject_line, subject_line_format, err):
    if False:
        while True:
            i = 10
    err.error('Subject line: ' + subject_line)
    if not subject_line.endswith('.'):
        err.error('* must end with "."')
    if not re.match('^[^!]+: ', subject_line):
        err.error('* must start with "path: "')
    if re.match('^[^!]+: *$', subject_line):
        err.error('* must contain a subject after the path.')
    m = re.match('^[^!]+: ([a-z][^ ]*)', subject_line)
    if m:
        err.error('* first word of subject ("{}") must be capitalised.'.format(m.group(1)))
    if re.match('^[^!]+: [^ ]+$', subject_line):
        err.error('* subject must contain more than one word.')
    err.error('* must match: ' + repr(subject_line_format))
    err.error('* Example: "py/runtime: Add support for foo to bar."')

def verify(sha, err):
    if False:
        return 10
    verbose('verify', sha)
    err.prefix = 'commit ' + sha + ': '
    for line in git_log('%ae%n%ce', sha, '-n1'):
        very_verbose('email', line)
        if 'noreply' in line:
            err.error('Unwanted email address: ' + line)
    raw_body = list(git_log('%B', sha, '-n1'))
    verify_message_body(raw_body, err)

def verify_message_body(raw_body, err):
    if False:
        print('Hello World!')
    if not raw_body:
        err.error('Message is empty')
        return
    subject_line = raw_body[0]
    for prefix in ignore_prefixes:
        if subject_line.startswith(prefix):
            verbose('Skipping ignored commit message')
            return
    very_verbose('subject_line', subject_line)
    subject_line_format = '^[^!]+: [A-Z]+.+ .+\\.$'
    if not re.match(subject_line_format, subject_line):
        diagnose_subject_line(subject_line, subject_line_format, err)
    if len(subject_line) >= 73:
        err.error('Subject line must be 72 or fewer characters: ' + subject_line)
    if len(raw_body) > 1 and raw_body[1]:
        err.error('Second message line must be empty: ' + raw_body[1])
    for line in raw_body[2:]:
        if len(line) >= 76 and '://' not in line:
            err.error('Message lines should be 75 or less characters: ' + line)
    if not raw_body[-1].startswith('Signed-off-by: ') or '@' not in raw_body[-1]:
        err.error('Message must be signed-off. Use "git commit -s".')

def run(args):
    if False:
        return 10
    verbose('run', *args)
    err = ErrorCollection()
    if '--check-file' in args:
        filename = args[-1]
        verbose('checking commit message from', filename)
        with open(args[-1]) as f:
            lines = [line.rstrip('\r\n') for line in f if not line.startswith('#')]
            while not lines[-1]:
                lines.pop()
            verify_message_body(lines, err)
    else:
        for sha in git_log('%h', *args):
            verify(sha, err)
    if err.has_errors or err.has_warnings:
        if suggestions:
            print('See https://github.com/micropython/micropython/blob/master/CODECONVENTIONS.md')
    else:
        print('ok')
    if err.has_errors:
        sys.exit(1)

def show_help():
    if False:
        while True:
            i = 10
    print('usage: verifygitlog.py [-v -n -h --check-file] ...')
    print('-v  : increase verbosity, can be specified multiple times')
    print('-n  : do not print multi-line suggestions')
    print('-h  : print this help message and exit')
    print('--check-file : Pass a single argument which is a file containing a candidate commit message')
    print('--ignore-rebase : Skip checking commits with git rebase autosquash prefixes or WIP as a prefix')
    print('... : arguments passed to git log to retrieve commits to verify')
    print('      see https://www.git-scm.com/docs/git-log')
    print('      passing no arguments at all will verify all commits')
    print('examples:')
    print('verifygitlog.py -n10  # Check last 10 commits')
    print('verifygitlog.py -v master..HEAD  # Check commits since master')
if __name__ == '__main__':
    args = sys.argv[1:]
    verbosity = args.count('-v')
    suggestions = args.count('-n') == 0
    if '--ignore-rebase' in args:
        args.remove('--ignore-rebase')
        ignore_prefixes = ['squash!', 'fixup!', 'amend!', 'WIP']
    if '-h' in args:
        show_help()
    else:
        args = [arg for arg in args if arg not in ['-v', '-n', '-h']]
        run(args)