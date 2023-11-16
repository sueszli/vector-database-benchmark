"""Handle ivre passiverecon2db files."""
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from argparse import ArgumentParser
from typing import Any, Dict, List, Match, Optional
from ivre import utils
SENSORS: Dict[str, str] = {}
FILEFORMAT = '^(?P<sensor>%s)[.-](?P<datetime>[0-9-]+)\\.log(?:\\.(?:gz|bz2))?$'
SLEEPTIME = 2
WANTDOWN = False

def shutdown(signum: int, _: Any) -> None:
    if False:
        i = 10
        return i + 15
    'Sets the global variable `WANTDOWN` to `True` to stop\n    everything after the current files have been processed.\n\n    '
    global WANTDOWN
    utils.LOGGER.info('SHUTDOWN: got signal %d, will halt after current file.', signum)
    WANTDOWN = True

def getnextfiles(directory: str, sensor: Optional[str]=None, count: int=1) -> List[Match[str]]:
    if False:
        while True:
            i = 10
    'Returns a list of maximum `count` filenames (as FILEFORMAT matches)\n    to process, given the `directory` and the `sensor` (or, if it is\n    `None`, from any sensor).\n\n    '
    if sensor is None:
        fmt = re.compile(FILEFORMAT % '[^\\.]*')
    else:
        fmt = re.compile(FILEFORMAT % re.escape(sensor))
    files = sorted((f for f in (fmt.match(f) for f in os.listdir(directory)) if f is not None), key=lambda x: [int(val) for val in x.groupdict()['datetime'].split('-')])
    return files[:count]

def create_process(progname: str, sensor: str) -> subprocess.Popen:
    if False:
        print('Hello World!')
    'Creates the insertion process for the given `sensor` using\n    `progname`.\n\n    '
    return subprocess.Popen(shlex.split(progname) + ['-s', SENSORS.get(sensor, sensor)], stdin=subprocess.PIPE)

def worker(progname: str, directory: str, sensor: Optional[str]=None) -> None:
    if False:
        print('Hello World!')
    'This function is the main loop, creating the processes when\n    needed and feeding them with the data from the files.\n\n    '
    utils.makedirs(os.path.join(directory, 'current'))
    procs: Dict[str, subprocess.Popen] = {}
    while not WANTDOWN:
        fname_l = getnextfiles(directory, sensor=sensor, count=1)
        if not fname_l:
            utils.LOGGER.debug('Sleeping for %d s', SLEEPTIME)
            time.sleep(SLEEPTIME)
            continue
        fname_m = fname_l[0]
        fname_sensor = fname_m.groupdict()['sensor']
        if fname_sensor in procs:
            proc = procs[fname_sensor]
        else:
            proc = create_process(progname, fname_sensor)
            procs[fname_sensor] = proc
        assert proc.stdin is not None
        fname = fname_m.group()
        try:
            shutil.move(os.path.join(directory, fname), os.path.join(directory, 'current'))
        except shutil.Error:
            continue
        utils.LOGGER.debug('Handling %s', fname)
        fname = os.path.join(directory, 'current', fname)
        fdesc = utils.open_file(fname)
        handled_ok = True
        for line in fdesc:
            try:
                proc.stdin.write(line)
            except ValueError:
                utils.LOGGER.warning('Error while handling line %r. Trying again', line)
                proc = create_process(progname, fname_sensor)
                assert proc.stdin is not None
                procs[fname_sensor] = proc
                try:
                    proc.stdin.write(line)
                    utils.LOGGER.warning('  ... OK')
                except ValueError:
                    handled_ok = False
                    utils.LOGGER.warning('  ... KO')
        fdesc.close()
        if handled_ok:
            os.unlink(fname)
            utils.LOGGER.debug('  ... OK')
        else:
            utils.LOGGER.debug('  ... KO')
    for proc in procs.values():
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait()

def main() -> None:
    if False:
        print('Hello World!')
    'Parses the arguments and call worker()'
    for s in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(s, shutdown)
        signal.siginterrupt(s, False)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--sensor', metavar='SENSOR[:SENSOR]', help='sensor to check, optionally with a long name, defaults to all.')
    parser.add_argument('--directory', metavar='DIR', help='base directory (defaults to /ivre/passiverecon/).', default='/ivre/passiverecon/')
    parser.add_argument('--progname', metavar='PROG', help='Program to run (defaults to ivre passiverecon2db).', default='ivre passiverecon2db')
    args = parser.parse_args()
    if args.sensor is not None:
        SENSORS.update(dict([args.sensor.split(':', 1) if ':' in args.sensor else [args.sensor, args.sensor]]))
        sensor = args.sensor.split(':', 1)[0]
    else:
        sensor = None
    worker(args.progname, args.directory, sensor=sensor)