import glob
import pathlib
import argparse

def create_dictionary(dataset_dir, save_dir):
    if False:
        while True:
            i = 10
    syllables = set()
    for p in pathlib.Path(dataset_dir).rglob('*.ssg'):
        with open(p) as f:
            sentences = f.readlines()
        for i in range(len(sentences)):
            sentences[i] = sentences[i].replace('\n', '')
            sentences[i] = sentences[i].replace('<s/>', '~')
            sentences[i] = sentences[i].split('~')
            syllables = syllables.union(sentences[i])
        print(len(syllables))
    import re
    a = []
    for s in syllables:
        print('---')
        if bool(re.match('^[\u0e00-\u0e7f]*$', s)) and s != '' and (' ' not in s):
            a.append(s)
        else:
            pass
    a = set(a)
    a = dict(zip(list(a), range(len(a))))
    import json
    print(a)
    print(len(a))
    with open(save_dir, 'w') as fp:
        json.dump(a, fp)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='syllable_segmentation_data', help='Directory for syllable dataset')
    parser.add_argument('--save_dir', type=str, default='thai-syllable.json', help='Directory for generated file')
    args = parser.parse_args()
    create_dictionary(args.dataset_dir, args.save_dir)