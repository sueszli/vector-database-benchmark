import shlex
import subprocess
import re

def parse_trades(stuff):
    if False:
        i = 10
        return i + 15
    '\n    >>> parse_trades("1 trades over 17 days (avg 0.06 trades/day)")\n    \'0.06\'\n    :param stuff:\n    :return:\n    '
    return stuff.split(b'avg')[-1].strip().split()[0]

def args_for_strategy(strat):
    if False:
        while True:
            i = 10
    ansi_escape = re.compile(b'\x1b[^m]*m')
    available = subprocess.check_output(shlex.split('env node ../../zenbot.js list-strategies'))
    strats = [ansi_escape.sub(b'', strat.strip()) for strat in available.split(b'\n\n')]
    groups = [group.splitlines() for group in strats]
    output = {split[0].split()[0]: split[1:] for split in groups if split}
    result = {strategy: [line.strip().strip(b'-').split(b'=')[0] for line in lines if b'--' in line] for (strategy, lines) in output.items()}
    result = {key.decode(): [p.decode() for p in val] for (key, val) in result.items()}
    return result[strat]

def strategies():
    if False:
        while True:
            i = 10
    ansi_escape = re.compile(b'\x1b[^m]*m')
    available = subprocess.check_output(shlex.split('env node ../../zenbot.js list-strategies'))
    strats = [ansi_escape.sub(b'', strat.strip()) for strat in available.split(b'\n\n')]
    groups = [group.splitlines() for group in strats]
    output = {split[0].split()[0]: split[1:] for split in groups if split}
    result = {strategy: [line.strip().strip(b'-').split(b'=')[0] for line in lines if b'--' in line] for (strategy, lines) in output.items()}
    result = [key.decode() for (key, val) in result.items()]
    return result