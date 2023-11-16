"""
Process the licensed version of SUC3 to BIO

The main program processes the expected location, or you can pass in a
specific zip or filename to read
"""
from io import TextIOWrapper
from zipfile import ZipFile

def extract(infile, outfile):
    if False:
        while True:
            i = 10
    '\n    Convert the infile to an outfile\n\n    Assumes the files are already open (this allows you to pass in a zipfile reader, for example)\n\n    The SUC3 format is like conll, but with the tags in tabs 10 and 11\n    '
    lines = infile.readlines()
    sentences = []
    cur_sentence = []
    for (idx, line) in enumerate(lines):
        line = line.strip()
        if not line:
            if cur_sentence:
                sentences.append(cur_sentence)
                cur_sentence = []
            continue
        pieces = line.split('\t')
        if len(pieces) < 12:
            raise ValueError('Unexpected line length in the SUC3 dataset at %d' % idx)
        if pieces[10] == 'O':
            cur_sentence.append((pieces[1], 'O'))
        else:
            cur_sentence.append((pieces[1], '%s-%s' % (pieces[10], pieces[11])))
    if cur_sentence:
        sentences.append(cur_sentence)
    for sentence in sentences:
        for word in sentence:
            outfile.write('%s\t%s\n' % word)
        outfile.write('\n')
    return len(sentences)

def extract_from_zip(zip_filename, in_filename, out_filename):
    if False:
        while True:
            i = 10
    '\n    Process a single file from SUC3\n\n    zip_filename: path to SUC3.0.zip\n    in_filename: which piece to read\n    out_filename: where to write the result\n    '
    with ZipFile(zip_filename) as zin:
        with zin.open(in_filename) as fin:
            with open(out_filename, 'w') as fout:
                num = extract(TextIOWrapper(fin, encoding='utf-8'), fout)
                print('Processed %d sentences from %s:%s to %s' % (num, zip_filename, in_filename, out_filename))
                return num

def process_suc3(zip_filename, short_name, out_dir):
    if False:
        i = 10
        return i + 15
    extract_from_zip(zip_filename, 'SUC3.0/corpus/conll/suc-train.conll', '%s/%s.train.bio' % (out_dir, short_name))
    extract_from_zip(zip_filename, 'SUC3.0/corpus/conll/suc-dev.conll', '%s/%s.dev.bio' % (out_dir, short_name))
    extract_from_zip(zip_filename, 'SUC3.0/corpus/conll/suc-test.conll', '%s/%s.test.bio' % (out_dir, short_name))

def main():
    if False:
        i = 10
        return i + 15
    process_suc3('extern_data/ner/sv_suc3/SUC3.0.zip', 'data/ner')
if __name__ == '__main__':
    main()