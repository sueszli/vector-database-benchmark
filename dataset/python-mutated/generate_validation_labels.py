"""Tool to convert ILSVRC devkit validation ground truth to synset labels."""
import argparse
from os import path
import sys
import scipy.io
_SYNSET_ARRAYS_RELATIVE_PATH = 'data/meta.mat'
_VALIDATION_FILE_RELATIVE_PATH = 'data/ILSVRC2012_validation_ground_truth.txt'

def _synset_to_word(filepath):
    if False:
        i = 10
        return i + 15
    'Returns synset to word dictionary by reading sysnset arrays.'
    mat = scipy.io.loadmat(filepath)
    entries = mat['synsets']
    fields = ['synset_id', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
    synset_index = fields.index('synset_id')
    words_index = fields.index('words')
    synset_to_word = {}
    for entry in entries:
        entry = entry[0]
        synset_id = int(entry[synset_index][0])
        first_word = entry[words_index][0].split(',')[0]
        synset_to_word[synset_id] = first_word
    return synset_to_word

def _validation_file_path(ilsvrc_dir):
    if False:
        for i in range(10):
            print('nop')
    return path.join(ilsvrc_dir, _VALIDATION_FILE_RELATIVE_PATH)

def _synset_array_path(ilsvrc_dir):
    if False:
        for i in range(10):
            print('nop')
    return path.join(ilsvrc_dir, _SYNSET_ARRAYS_RELATIVE_PATH)

def _generate_validation_labels(ilsvrc_dir, output_file):
    if False:
        return 10
    synset_to_word = _synset_to_word(_synset_array_path(ilsvrc_dir))
    with open(_validation_file_path(ilsvrc_dir), 'r') as synset_id_file, open(output_file, 'w') as output:
        for synset_id in synset_id_file:
            synset_id = int(synset_id)
            output.write('%s\n' % synset_to_word[synset_id])

def _check_arguments(args):
    if False:
        i = 10
        return i + 15
    if not args.validation_labels_output:
        raise ValueError('Invalid path to output file.')
    ilsvrc_dir = args.ilsvrc_devkit_dir
    if not ilsvrc_dir or not path.isdir(ilsvrc_dir):
        raise ValueError('Invalid path to ilsvrc_dir')
    if not path.exists(_validation_file_path(ilsvrc_dir)):
        raise ValueError('Invalid path to ilsvrc_dir, cannot find validation file.')
    if not path.exists(_synset_array_path(ilsvrc_dir)):
        raise ValueError('Invalid path to ilsvrc_dir, cannot find synset arrays file.')

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Converts ILSVRC devkit validation_ground_truth.txt to synset labels file that can be used by the accuracy script.')
    parser.add_argument('--validation_labels_output', type=str, help='Full path for outputting validation labels.')
    parser.add_argument('--ilsvrc_devkit_dir', type=str, help='Full path to ILSVRC 2012 devkit directory.')
    args = parser.parse_args()
    try:
        _check_arguments(args)
    except ValueError as e:
        parser.print_usage()
        file_name = path.basename(sys.argv[0])
        sys.stderr.write('{0}: error: {1}\n'.format(file_name, str(e)))
        sys.exit(1)
    _generate_validation_labels(args.ilsvrc_devkit_dir, args.validation_labels_output)
if __name__ == '__main__':
    main()