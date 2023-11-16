import subprocess
import click

def test(cmd, limit):
    if False:
        while True:
            i = 10
    print(f'Testing PYTORCH_JIT_OPT_LIMIT=tensorexpr_fuser={limit} {cmd}')
    p = subprocess.run(f'PYTORCH_JIT_OPT_LIMIT=tensorexpr_fuser={limit} {cmd}', shell=True, capture_output=True, encoding='utf-8', check=False)
    print(p.stdout)
    f = 'INTERNAL ASSERT FAILED'
    if f in p.stdout or f in p.stderr:
        print('skip')
        return -1
    if p.returncode == 0:
        print('good')
        return 1
    print('bad')
    return 0

@click.command()
@click.option('--cmd')
def bisect(cmd):
    if False:
        i = 10
        return i + 15
    last_good = 0
    first_bad = 10000
    skips = set()

    def keep_going():
        if False:
            return 10
        for limit in range(last_good + 1, first_bad):
            if limit not in skips:
                return True
        return False
    while keep_going():
        test_limit = test_mid = (last_good + first_bad) // 2
        val = -1
        while test_limit <= first_bad and val == -1:
            val = test(cmd, test_limit)
            if val == -1:
                skips.add(test_limit)
                test_limit = test_limit + 1
        if val == -1:
            test_limit = test_mid - 1
            while test_limit >= last_good and val == -1:
                val = test(cmd, test_limit)
                if val == -1:
                    skips.add(test_limit)
                    test_limit = test_limit - 1
        if val == 0:
            first_bad = test_limit
        elif val == 1:
            last_good = test_limit
    print(f'last good: {last_good}, first bad: {first_bad}')
if __name__ == '__main__':
    bisect()