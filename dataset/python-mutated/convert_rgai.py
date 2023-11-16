"""
This script converts the Hungarian files available at u-szeged
  https://rgai.inf.u-szeged.hu/node/130
"""
import os
import tempfile
from stanza.utils.datasets.ner.split_wikiner import split_wikiner

def read_rgai_file(filename, separator):
    if False:
        for i in range(10):
            print('nop')
    with open(filename, encoding='latin-1') as fin:
        lines = fin.readlines()
        lines = [x.strip() for x in lines]
        for (idx, line) in enumerate(lines):
            if not line:
                continue
            pieces = lines[idx].split(separator)
            if len(pieces) != 2:
                raise ValueError('Line %d is in an unexpected format!  Expected exactly two pieces when split on %s' % (idx, separator))
            if pieces[-1] == '0':
                pieces[-1] = 'O'
                lines[idx] = '\t'.join(pieces)
    print('Read %d lines from %s' % (len(lines), filename))
    return lines

def get_rgai_data(base_input_path, use_business, use_criminal):
    if False:
        return 10
    assert use_business or use_criminal, 'Must specify one or more sections of the dataset to use'
    dataset_lines = []
    if use_business:
        business_file = os.path.join(base_input_path, 'hun_ner_corpus.txt')
        lines = read_rgai_file(business_file, '\t')
        dataset_lines.extend(lines)
    if use_criminal:
        criminal_file = os.path.join(base_input_path, 'HVGJavNENoContext')
        lines = read_rgai_file(criminal_file, ' ')
        dataset_lines.extend(lines)
    return dataset_lines

def convert_rgai(base_input_path, base_output_path, short_name, use_business, use_criminal):
    if False:
        print('Hello World!')
    all_data_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        raw_data = get_rgai_data(base_input_path, use_business, use_criminal)
        for line in raw_data:
            all_data_file.write(line.encode())
            all_data_file.write('\n'.encode())
        all_data_file.close()
        split_wikiner(base_output_path, all_data_file.name, prefix=short_name)
    finally:
        os.unlink(all_data_file.name)