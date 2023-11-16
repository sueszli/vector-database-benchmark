from __future__ import print_function
import re
import sys
from prospector.run import main

def _excepthook(e, t, tb):
    if False:
        for i in range(10):
            print('nop')
    while tb is not None:
        frame = tb.tb_frame
        print(frame.f_code, frame.f_code.co_name)
        for n in ('self', 'node', 'elt'):
            if n in frame.f_locals:
                print(n, frame.f_locals[n])
        print('---')
        tb = tb.tb_next
sys.excepthook = _excepthook
if __name__ == '__main__':
    sys.argv[0] = re.sub('(-script\\.pyw|\\.exe)?$', '', sys.argv[0])
    sys.exit(main())