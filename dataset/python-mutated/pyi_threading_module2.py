import os
import sys
import threading
_OUT_EXPECTED = ['ONE', 'TWO', 'THREE']
if 'PYI_THREAD_TEST_CASE' in os.environ:

    class TestThreadClass(threading.Thread):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            threading.Thread.__init__(self)

        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            print('ONE')
            print('TWO')
            print('THREE')
    TestThreadClass().start()
else:
    itself = sys.argv[0]
    import subprocess
    env = dict(os.environ)
    env['PYI_THREAD_TEST_CASE'] = 'yes'
    proc = subprocess.Popen([itself], stdout=subprocess.PIPE, env=env, stderr=subprocess.PIPE, shell=False)
    (out, err) = proc.communicate()
    print(out)
    out = out.decode('ascii')
    print(out)
    out = out.strip().splitlines()
    for line in out:
        if not line.strip():
            out.remove(line)
    if out != _OUT_EXPECTED:
        print(' +++++++ SUBPROCESS ERROR OUTPUT +++++++')
        print(err)
        raise SystemExit('Subprocess did not print ONE, TWO, THREE in correct order. (output was %r, return code was %s)' % (out, proc.returncode))