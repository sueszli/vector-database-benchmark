import argparse
from fairseq.data import Dictionary, data_utils, indexed_dataset

def get_parser():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='writes text from binarized file to stdout')
    parser.add_argument('--dataset-impl', help='dataset implementation', choices=indexed_dataset.get_available_dataset_impl())
    parser.add_argument('--dict', metavar='FP', help='dictionary containing known words', default=None)
    parser.add_argument('--input', metavar='FP', required=True, help='binarized file to read')
    return parser

def main():
    if False:
        i = 10
        return i + 15
    parser = get_parser()
    args = parser.parse_args()
    dictionary = Dictionary.load(args.dict) if args.dict is not None else None
    dataset = data_utils.load_indexed_dataset(args.input, dictionary, dataset_impl=args.dataset_impl, default='lazy')
    for tensor_line in dataset:
        if dictionary is None:
            line = ' '.join([str(int(x)) for x in tensor_line])
        else:
            line = dictionary.string(tensor_line)
        print(line)
if __name__ == '__main__':
    main()