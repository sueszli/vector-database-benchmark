import argparse
import os
import sys
import textwrap
import pandas as pd

def get_field(csv, model_name: str, field: str):
    if False:
        print('Hello World!')
    try:
        return csv.loc[csv['name'] == model_name][field].item()
    except Exception as e:
        return None

def check_accuracy(actual_csv, expected_csv, expected_filename):
    if False:
        i = 10
        return i + 15
    failed = []
    improved = []
    for model in actual_csv['name']:
        accuracy = get_field(actual_csv, model, 'accuracy')
        expected_accuracy = get_field(expected_csv, model, 'accuracy')
        if accuracy == expected_accuracy:
            status = 'PASS' if expected_accuracy == 'pass' else 'XFAIL'
            print(f'{model:34}  {status}')
            continue
        elif accuracy != 'pass':
            status = 'FAIL:'
            failed.append(model)
        else:
            status = 'IMPROVED:'
            improved.append(model)
        print(f'{model:34}  {status:9} accuracy={accuracy}, expected={expected_accuracy}')
    msg = ''
    if failed or improved:
        if failed:
            msg += textwrap.dedent(f"\n            Error: {len(failed)} models have accuracy status regressed:\n                {' '.join(failed)}\n\n            ")
        if improved:
            msg += textwrap.dedent(f"\n            Improvement: {len(improved)} models have accuracy status improved:\n                {' '.join(improved)}\n\n            ")
        sha = os.getenv('SHA1', '{your CI commit sha}')
        msg += textwrap.dedent(f'\n        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.\n        from pytorch/pytorch root, run\n        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`\n        and then `git add` the resulting local changes to expected CSVs to your commit.\n        ')
    return (failed or improved, msg)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual', type=str, required=True)
    parser.add_argument('--expected', type=str, required=True)
    args = parser.parse_args()
    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)
    (failed, msg) = check_accuracy(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)
if __name__ == '__main__':
    main()