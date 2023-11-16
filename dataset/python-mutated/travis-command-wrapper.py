import errno
import optparse
import signal
import subprocess
import sys
import time
parser = optparse.OptionParser()
parser.add_option('-s', '--status', help='invoked in a shell when it is time for a status report', default='date; free; df -h .; COLUMNS=200 LINES=30 top -w -b -n 1 2>/dev/null || top -n 1; ps x --cols 200 --forest', metavar='SHELL-CMD')
parser.add_option('-i', '--interval', help='repeat status at intervals of this amount of seconds, 0 to disable', default=300, metavar='SECONDS', type='int')
parser.add_option('-d', '--deadline', help='stop execution when reaching the given time', default=time.time, metavar='SECONDS-SINCE-EPOCH', type='int')
(options, args) = parser.parse_args()

def check_deadline(now):
    if False:
        i = 10
        return i + 15
    if options.deadline > 0 and options.deadline < now:
        print('\n\n*** travis-cmd-wrapper: deadline reached, shutting down ***\n\n')
        sys.exit(1)
    else:
        print('deadline not reached: %s > %s' % (options.deadline, now))
now = time.time()
next_status = now + options.interval
alarm_interval = max(options.interval, 0)
if options.deadline:
    check_deadline(now)
    if options.deadline < now + 60:
        deadline_alarm_interval = max(int(options.deadline + 2 - now), 1)
    elif next_status > 60:
        deadline_alarm_interval = 60
    if deadline_alarm_interval < alarm_interval:
        alarm_interval = deadline_alarm_interval

def status(signum, frame):
    if False:
        while True:
            i = 10
    global next_status
    now = time.time()
    if options.interval < 0 or now >= next_status:
        subprocess.call(options.status, shell=True)
        next_status = now + options.interval
    check_deadline(now)
    if alarm_interval > 0:
        signal.alarm(alarm_interval)
try:
    cmd = subprocess.Popen(args, stdout=subprocess.PIPE)
    signal.signal(signal.SIGALRM, status)
    if alarm_interval > 0:
        signal.alarm(alarm_interval)
    while cmd.poll() is None:
        try:
            line = cmd.stdout.readline()
            sys.stdout.write(line)
            sys.stdout.flush()
        except IOError as ex:
            if ex.errno != errno.EINTR:
                raise
finally:
    if cmd.poll() is None:
        cmd.kill()
exit(1 if cmd.returncode else 0)