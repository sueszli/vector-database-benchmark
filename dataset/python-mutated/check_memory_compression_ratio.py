import argparse
import sys
import textwrap
import pandas as pd

def main(args):
    if False:
        for i in range(10):
            print('nop')
    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)
    failed = []
    for name in actual['name']:
        actual_memory_compression = float(actual.loc[actual['name'] == name]['compression_ratio'])
        try:
            expected_memory_compression = float(expected.loc[expected['name'] == name]['compression_ratio'])
        except TypeError:
            print(f'{name:34} is missing from {args.expected}')
            continue
        if actual_memory_compression >= expected_memory_compression * 0.95:
            status = 'PASS'
        else:
            status = 'FAIL'
            failed.append(name)
        print(f'\n            {name:34}:\n                actual_memory_compression={actual_memory_compression:.2f},\n                expected_memory_compression={expected_memory_compression:.2f},\n                {status}\n            ')
    if failed:
        print(textwrap.dedent(f"\n                Error: {len(failed)} models below expected memory compression ratio:\n                    {' '.join(failed)}\n                If this drop is expected, you can update `{args.expected}`.\n                "))
        sys.exit(1)
parser = argparse.ArgumentParser()
parser.add_argument('--actual', type=str, required=True)
parser.add_argument('--expected', type=str, required=True)
args = parser.parse_args()
if __name__ == '__main__':
    main(args)