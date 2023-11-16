import argparse
import sys
import textwrap
import pandas as pd

def check_perf_csv(filename, threshold):
    if False:
        while True:
            i = 10
    '\n    Basic performance checking.\n    '
    df = pd.read_csv(filename)
    failed = []
    for (_, row) in df.iterrows():
        model_name = row['name']
        speedup = row['speedup']
        if speedup < threshold:
            failed.append(model_name)
        print(f'{model_name:34} {speedup}')
    if failed:
        print(textwrap.dedent(f"\n                Error {len(failed)} models performance regressed\n                    {' '.join(failed)}\n                "))
        sys.exit(1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='csv file name')
    parser.add_argument('--threshold', '-t', type=float, help='threshold speedup value to check against')
    args = parser.parse_args()
    check_perf_csv(args.file, args.threshold)