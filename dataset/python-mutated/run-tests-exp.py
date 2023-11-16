import sys
import os
tests = ['basics', 'micropython', 'float', 'import', 'io', ' misc', 'unicode', 'extmod', 'unix']
if sys.platform == 'win32':
    MICROPYTHON = 'micropython.exe'
else:
    MICROPYTHON = 'micropython'

def should_skip(test):
    if False:
        print('Hello World!')
    if test.startswith('native'):
        return True
    if test.startswith('viper'):
        return True
test_count = 0
passed_count = 0
skip_count = 0
for suite in tests:
    if sys.platform == 'win32':
        r = os.system('dir /b %s/*.py >tests.lst' % suite)
    else:
        r = os.system('ls %s/*.py | xargs -n1 basename >tests.lst' % suite)
    assert r == 0
    with open('tests.lst') as f:
        testcases = f.readlines()
        testcases = [l[:-1] for l in testcases]
    assert testcases, "No tests found in dir '%s', which is implausible" % suite
    for t in testcases:
        if t == 'native_check.py':
            continue
        qtest = '%s/%s' % (suite, t)
        if should_skip(t):
            print('skip ' + qtest)
            skip_count += 1
            continue
        exp = None
        try:
            f = open(qtest + '.exp')
            exp = f.read()
            f.close()
        except OSError:
            pass
        if exp is not None:
            r = os.system(MICROPYTHON + ' %s >.tst.out' % qtest)
            if r == 0:
                f = open('.tst.out')
                out = f.read()
                f.close()
            else:
                out = 'CRASH'
            if out == 'SKIP\n':
                print('skip ' + qtest)
                skip_count += 1
            else:
                if out == exp:
                    print('pass ' + qtest)
                    passed_count += 1
                else:
                    print('FAIL ' + qtest)
                test_count += 1
        else:
            skip_count += 1
print('%s tests performed' % test_count)
print('%s tests passed' % passed_count)
if test_count != passed_count:
    print('%s tests failed' % (test_count - passed_count))
if skip_count:
    print('%s tests skipped' % skip_count)