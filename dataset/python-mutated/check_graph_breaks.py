import argparse
import os
import sys
import textwrap
import pandas as pd

def get_field(csv, model_name: str, field: str):
    if False:
        return 10
    try:
        return csv.loc[csv['name'] == model_name][field].item()
    except Exception as e:
        return None

def check_graph_breaks(actual_csv, expected_csv, expected_filename):
    if False:
        while True:
            i = 10
    failed = []
    improved = []
    for model in actual_csv['name']:
        graph_breaks = get_field(actual_csv, model, 'graph_breaks')
        expected_graph_breaks = get_field(expected_csv, model, 'graph_breaks')
        if graph_breaks == expected_graph_breaks:
            status = 'PASS'
            print(f'{model:34}  {status}')
            continue
        elif graph_breaks > expected_graph_breaks:
            status = 'FAIL:'
            failed.append(model)
        elif graph_breaks < expected_graph_breaks:
            status = 'IMPROVED:'
            improved.append(model)
        print(f'{model:34}  {status:9} graph_breaks={graph_breaks}, expected={expected_graph_breaks}')
    msg = ''
    if failed or improved:
        if failed:
            msg += textwrap.dedent(f"\n            Error: {len(failed)} models have new dynamo graph breaks:\n                {' '.join(failed)}\n\n            ")
        if improved:
            msg += textwrap.dedent(f"\n            Improvement: {len(improved)} models have fixed dynamo graph breaks:\n                {' '.join(improved)}\n\n            ")
        sha = os.getenv('SHA1', '{your CI commit sha}')
        msg += textwrap.dedent(f'\n        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.\n        from pytorch/pytorch root, run\n        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`\n        and then `git add` the resulting local changes to expected CSVs to your commit.\n        ')
    return (failed or improved, msg)

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual', type=str, required=True)
    parser.add_argument('--expected', type=str, required=True)
    args = parser.parse_args()
    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)
    (failed, msg) = check_graph_breaks(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)
if __name__ == '__main__':
    main()