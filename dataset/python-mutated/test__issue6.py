from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import sys
if not sys.argv[1:]:
    from subprocess import Popen, PIPE
    p = Popen([sys.executable, __file__, 'subprocess'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate(b'hello world\n')
    code = p.poll()
    assert p.poll() == 0, (out, err, code)
    assert out.strip() == b'11 chars.', (out, err, code)
    assert err == b'' or b'sys.excepthook' in err or b'Warning' in err, (out, err, code)
elif sys.argv[1:] == ['subprocess']:
    import gevent
    import gevent.monkey
    gevent.monkey.patch_all(sys=True)

    def printline():
        if False:
            print('Hello World!')
        try:
            line = raw_input()
        except NameError:
            line = input()
        print('%s chars.' % len(line))
        sys.stdout.flush()
    gevent.spawn(printline).join()
else:
    sys.exit('Invalid arguments: %r' % (sys.argv,))