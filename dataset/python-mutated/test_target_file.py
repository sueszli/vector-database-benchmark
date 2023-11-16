import subprocess

def test_target_file():
    if False:
        for i in range(10):
            print('nop')
    '\n        Check that output of passing in file is the same\n        as having that file in -target-file flag\n    '
    output = subprocess.check_output(['bin/semgrep-core', '-e', '$X==$X', '-lang', 'python', 'tests/semgrep-core-e2e/targets/basic.py'], encoding='utf-8')
    output2 = subprocess.check_output(['bin/semgrep-core', '-e', '$X==$X', '-lang', 'python', '-targets', 'tests/semgrep-core-e2e/targets.json'], encoding='utf-8')
    assert output == output2
if __name__ == '__main__':
    test_target_file()