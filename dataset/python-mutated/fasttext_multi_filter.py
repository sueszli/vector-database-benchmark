import fasttext
from multiprocessing import Pool
import contextlib
import sys
import argparse
from functools import partial
import io
model = None

def init(model_path):
    if False:
        return 10
    global model
    model = fasttext.load_model(model_path)

def pred(lines):
    if False:
        for i in range(10):
            print('nop')
    return (lines, [model.predict(line.strip())[0][0][9:] for line in lines])

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model to load')
    parser.add_argument('--inputs', nargs='+', default=['-'], help='input files to filter')
    parser.add_argument('--langs', nargs='+', required=True, help='lang ids of each input file')
    parser.add_argument('--outputs', nargs='+', default=['-'], help='path to save lid filtered outputs')
    parser.add_argument('--num-workers', type=int, metavar='N', default=10, help='number of processes in parallel')
    args = parser.parse_args()
    assert len(args.inputs) == len(args.langs) and len(args.inputs) == len(args.outputs)
    with contextlib.ExitStack() as stack:
        inputs = [stack.enter_context(open(input, 'r', encoding='utf-8', newline='\n', errors='replace')) if input != '-' else io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace') for input in args.inputs]
        outputs = [stack.enter_context(open(output, 'w', encoding='utf-8', newline='\n')) if output != '-' else sys.stdout for output in args.outputs]
        with Pool(args.num_workers, initializer=partial(init, args.model)) as p:
            skip_cnt = 0
            for (lines, preds) in p.imap(pred, list(zip(*inputs)), chunksize=500):
                if not all((a == b for (a, b) in zip(preds, args.langs))):
                    skip_cnt += 1
                    continue
                for (line, output_h) in zip(lines, outputs):
                    print(line.strip(), file=output_h)
        print(f'Skipped {skip_cnt} lines.')
if __name__ == '__main__':
    main()