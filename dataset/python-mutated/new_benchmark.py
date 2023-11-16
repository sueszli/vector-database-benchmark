import json
import subprocess
import re
import statistics
from glob import glob
from collections import defaultdict
import sys
cmds = {'yappi_wall': ['python3', '-m', 'yappi', '-o', '/dev/null', '-c', 'wall'], 'yappi_cpu': ['python3', '-m', 'yappi', '-o', '/dev/null', '-c', 'cpu']}
result_regexp = re.compile('Time elapsed:\\s+([0-9]*\\.[0-9]+)')

def main():
    if False:
        return 10
    out = defaultdict(lambda : {})
    for progname in ['./test/expensive_benchmarks/bm_docutils.py']:
        for (profile_name, profile_cmd) in cmds.items():
            times = []
            for i in range(5):
                print(f'''Running {profile_name} on {progname} using "{' '.join(profile_cmd + progname.split(' '))}"...''', end='', flush=True)
                result = subprocess.run(profile_cmd + progname.split(' '), stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
                output = result.stdout.decode('utf-8')
                match = result_regexp.search(output)
                if match is not None:
                    print(f'... {match.group(1)}', end='\n' if profile_name != 'memray' else '')
                    times.append(round(100 * float(match.group(1))) / 100.0)
                    if profile_name == 'memray':
                        res2 = subprocess.run(['time', sys.executable, '-m', 'memray', 'flamegraph', '-f', '/tmp/memray.out'], capture_output=True, env={'TIME': 'Time elapsed: %e'})
                        output2 = res2.stderr.decode('utf-8')
                        match2 = result_regexp.search(output2)
                        if match2 is not None:
                            print(f'... {match2.group(1)}')
                            times[-1] += round(100 * float(match2.group(1))) / 100.0
                        else:
                            print('... RUN FAILED')
                else:
                    print('RUN FAILED')
            out[profile_name][progname] = times
    with open('yappi.json', 'w+') as f:
        json.dump(dict(out), f)
if __name__ == '__main__':
    main()