import argparse
import os

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='test script')
    parser.add_argument('--local-rank', '--local_rank', type=int, required=True, help='The rank of the node for multi-node distributed training')
    return parser.parse_args()

def main():
    if False:
        while True:
            i = 10
    print('Start execution')
    args = parse_args()
    expected_rank = int(os.environ['LOCAL_RANK'])
    actual_rank = args.local_rank
    if expected_rank != actual_rank:
        raise RuntimeError(f'Parameters passed: --local-rank that has different value from env var: expected: {expected_rank}, got: {actual_rank}')
    print('End execution')
if __name__ == '__main__':
    main()