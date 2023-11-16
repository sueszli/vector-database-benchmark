import os
import sys
import time
from hashlib import md5
from buildbot.util import unicode2bytes

def tryserver(config):
    if False:
        for i in range(10):
            print('nop')
    jobdir = os.path.expanduser(config['jobdir'])
    job = sys.stdin.read()
    timestring = f'{time.time()}'
    m = md5()
    job = unicode2bytes(job)
    m.update(job)
    jobhash = m.hexdigest()
    fn = f'{timestring}-{jobhash}'
    tmpfile = os.path.join(jobdir, 'tmp', fn)
    newfile = os.path.join(jobdir, 'new', fn)
    with open(tmpfile, 'wb') as f:
        f.write(job)
    os.rename(tmpfile, newfile)
    return 0