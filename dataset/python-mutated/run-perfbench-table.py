import os
import subprocess
import sys
import time
import argparse
from glob import glob
from rich.live import Live
from rich.console import Console
from rich.table import Table
sys.path.append('../tools')
import pyboard
if os.name == 'nt':
    CPYTHON3 = os.getenv('MICROPY_CPYTHON3', 'python3.exe')
    MICROPYTHON = os.getenv('MICROPY_MICROPYTHON', '../ports/windows/micropython.exe')
else:
    CPYTHON3 = os.getenv('MICROPY_CPYTHON3', 'python3')
    MICROPYTHON = os.getenv('MICROPY_MICROPYTHON', '../ports/unix/micropython')
PYTHON_TRUTH = CPYTHON3
BENCH_SCRIPT_DIR = 'perf_bench/'

def compute_stats(lst):
    if False:
        for i in range(10):
            print('nop')
    avg = 0
    var = 0
    for x in lst:
        avg += x
        var += x * x
    avg /= len(lst)
    var = max(0, var / len(lst) - avg ** 2)
    return (avg, var ** 0.5)

def run_script_on_target(target, script, run_command=None):
    if False:
        print('Hello World!')
    output = b''
    err = None
    if isinstance(target, pyboard.Pyboard):
        try:
            target.enter_raw_repl()
            start_ts = time.monotonic_ns()
            output = target.exec_(script)
            if run_command:
                start_ts = time.monotonic_ns()
                output = target.exec_(run_command)
            end_ts = time.monotonic_ns()
        except pyboard.PyboardError as er:
            end_ts = time.monotonic_ns()
            err = er
        finally:
            target.exit_raw_repl()
    else:
        try:
            if run_command:
                script += run_command
            start_ts = time.monotonic_ns()
            p = subprocess.run(target, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, input=script)
            end_ts = time.monotonic_ns()
            output = p.stdout
        except subprocess.CalledProcessError as er:
            end_ts = time.monotonic_ns()
            err = er
    return (str(output.strip(), 'ascii'), err, (end_ts - start_ts) // 1000)

def run_feature_test(target, test):
    if False:
        i = 10
        return i + 15
    with open('feature_check/' + test + '.py', 'rb') as f:
        script = f.read()
    (output, err, _) = run_script_on_target(target, script)
    if err is None:
        return output
    else:
        return 'CRASH: %r' % err

def run_benchmark_on_target(target, script, run_command=None):
    if False:
        print('Hello World!')
    (output, err, runtime_us) = run_script_on_target(target, script, run_command)
    if err is None:
        (time, norm, result) = output.split(None, 2)
        try:
            return (int(time), int(norm), result, runtime_us)
        except ValueError:
            return (-1, -1, 'CRASH: %r' % output, runtime_us)
    else:
        return (-1, -1, 'CRASH: %r' % err, runtime_us)

def run_benchmarks(console, target, param_n, param_m, n_average, test_list):
    if False:
        return 10
    skip_complex = run_feature_test(target, 'complex') != 'complex'
    skip_native = run_feature_test(target, 'native_check') != 'native'
    table = Table(show_header=True)
    table.add_column('Test')
    table.add_column('Time', justify='right')
    table.add_column('Score', justify='right')
    table.add_column('Ref Time', justify='right')
    live = Live(table, console=console)
    live.start()
    for test_file in sorted(test_list):
        skip = skip_complex and test_file.find('bm_fft') != -1 or (skip_native and test_file.find('viper_') != -1)
        if skip:
            print('skip')
            table.add_row(test_file, *['skip'] * 6)
            continue
        with open(test_file, 'rb') as f:
            test_script = f.read()
        with open(BENCH_SCRIPT_DIR + 'benchrun.py', 'rb') as f:
            test_script += f.read()
        bm_run = b'bm_run(%u, %u)\n' % (param_n, param_m)
        if 0:
            with open('%s.full' % test_file, 'wb') as f:
                f.write(test_script)
        times = []
        runtimes = []
        scores = []
        error = None
        result_out = None
        for _ in range(n_average):
            (self_time, norm, result, runtime_us) = run_benchmark_on_target(target, test_script, bm_run)
            if self_time < 0 or norm < 0:
                error = result
                break
            if result_out is None:
                result_out = result
            elif result != result_out:
                error = 'FAIL self'
                break
            times.append(self_time)
            runtimes.append(runtime_us)
            scores.append(1000000.0 * norm / self_time)
        if error is None and result_out != 'None':
            (_, _, result_exp, _) = run_benchmark_on_target(PYTHON_TRUTH, test_script, bm_run)
            if result_out != result_exp:
                error = 'FAIL truth'
        if error is not None:
            print(test_file, error)
            if error == 'no matching params':
                table.add_row(test_file, *[None] * 3)
            else:
                table.add_row(test_file, *['error'] * 3)
        else:
            (t_avg, t_sd) = compute_stats(times)
            (r_avg, r_sd) = compute_stats(runtimes)
            (s_avg, s_sd) = compute_stats(scores)
            table.add_row(test_file, f'{t_avg:.2f}±{100 * t_sd / t_avg:.1f}%', f'{s_avg:.2f}±{100 * s_sd / s_avg:.1f}%', f'{r_avg:.2f}±{100 * r_sd / r_avg:.1f}%')
            if 0:
                print('  times: ', times)
                print('  scores:', scores)
        live.update(table, refresh=True)
    live.stop()

def parse_output(filename):
    if False:
        for i in range(10):
            print('nop')
    with open(filename) as f:
        params = f.readline()
        (n, m, _) = params.strip().split()
        n = int(n.split('=')[1])
        m = int(m.split('=')[1])
        data = []
        for l in f:
            if l.find(': ') != -1 and l.find(': skip') == -1 and (l.find('CRASH: ') == -1):
                (name, values) = l.strip().split(': ')
                values = tuple((float(v) for v in values.split()))
                data.append((name,) + values)
    return (n, m, data)

def compute_diff(file1, file2, diff_score):
    if False:
        for i in range(10):
            print('nop')
    (n1, m1, d1) = parse_output(file1)
    (n2, m2, d2) = parse_output(file2)
    if diff_score:
        print('diff of scores (higher is better)')
    else:
        print('diff of microsecond times (lower is better)')
    if n1 == n2 and m1 == m2:
        hdr = 'N={} M={}'.format(n1, m1)
    else:
        hdr = 'N={} M={} vs N={} M={}'.format(n1, m1, n2, m2)
    print('{:24} {:>10} -> {:>10}   {:>10}   {:>7}% (error%)'.format(hdr, file1, file2, 'diff', 'diff'))
    while d1 and d2:
        if d1[0][0] == d2[0][0]:
            entry1 = d1.pop(0)
            entry2 = d2.pop(0)
            name = entry1[0].rsplit('/')[-1]
            (av1, sd1) = (entry1[1 + 2 * diff_score], entry1[2 + 2 * diff_score])
            (av2, sd2) = (entry2[1 + 2 * diff_score], entry2[2 + 2 * diff_score])
            sd1 *= av1 / 100
            sd2 *= av2 / 100
            av_diff = av2 - av1
            sd_diff = (sd1 ** 2 + sd2 ** 2) ** 0.5
            percent = 100 * av_diff / av1
            percent_sd = 100 * sd_diff / av1
            print('{:24} {:10.2f} -> {:10.2f} : {:+10.2f} = {:+7.3f}% (+/-{:.2f}%)'.format(name, av1, av2, av_diff, percent, percent_sd))
        elif d1[0][0] < d2[0][0]:
            d1.pop(0)
        else:
            d2.pop(0)

def main():
    if False:
        for i in range(10):
            print('nop')
    cmd_parser = argparse.ArgumentParser(description='Run benchmarks for MicroPython')
    cmd_parser.add_argument('-t', '--diff-time', action='store_true', help='diff time outputs from a previous run')
    cmd_parser.add_argument('-s', '--diff-score', action='store_true', help='diff score outputs from a previous run')
    cmd_parser.add_argument('-p', '--pyboard', action='store_true', help='run tests via pyboard.py')
    cmd_parser.add_argument('-d', '--device', default='/dev/ttyACM0', help='the device for pyboard.py')
    cmd_parser.add_argument('-a', '--average', default='8', help='averaging number')
    cmd_parser.add_argument('--emit', default='bytecode', help='MicroPython emitter to use (bytecode or native)')
    cmd_parser.add_argument('N', nargs=1, help='N parameter (approximate target CPU frequency)')
    cmd_parser.add_argument('M', nargs=1, help='M parameter (approximate target heap in kbytes)')
    cmd_parser.add_argument('files', nargs='*', help='input test files')
    args = cmd_parser.parse_args()
    if args.diff_time or args.diff_score:
        compute_diff(args.N[0], args.M[0], args.diff_score)
        sys.exit(0)
    N = int(args.N[0])
    M = int(args.M[0])
    n_average = int(args.average)
    if args.pyboard:
        target = pyboard.Pyboard(args.device)
        target.enter_raw_repl()
    else:
        target = [MICROPYTHON, '-X', 'emit=' + args.emit]
    if len(args.files) == 0:
        tests_skip = ('benchrun.py',)
        if M <= 25:
            tests_skip += ('bm_chaos.py', 'bm_hexiom.py', 'misc_raytrace.py')
        tests = sorted((BENCH_SCRIPT_DIR + test_file for test_file in os.listdir(BENCH_SCRIPT_DIR) if test_file.endswith('.py') and test_file not in tests_skip))
    else:
        tests = sorted(args.files)
    console = Console()
    print('N={} M={} n_average={}'.format(N, M, n_average))
    run_benchmarks(console, target, N, M, n_average, tests)
    if isinstance(target, pyboard.Pyboard):
        target.exit_raw_repl()
        target.close()
if __name__ == '__main__':
    main()