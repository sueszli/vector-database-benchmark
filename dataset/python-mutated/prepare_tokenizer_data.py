import argparse
import json
import os
import re
import sys
from collections import Counter
'\nData is output in 4 files:\n\na file containing the mwt information\na file containing the words and sentences in conllu format\na file containing the raw text of each paragraph\na file of 0,1,2 indicating word break or sentence break on a character level for the raw text\n  1: end of word\n  2: end of sentence\n'
PARAGRAPH_BREAK = re.compile('\\n\\s*\\n')

def is_para_break(index, text):
    if False:
        i = 10
        return i + 15
    ' Detect if a paragraph break can be found, and return the length of the paragraph break sequence. '
    if text[index] == '\n':
        para_break = PARAGRAPH_BREAK.match(text, index)
        if para_break:
            break_len = len(para_break.group(0))
            return (True, break_len)
    return (False, 0)

def find_next_word(index, text, word, output):
    if False:
        return 10
    '\n    Locate the next word in the text. In case a paragraph break is found, also write paragraph break to labels.\n    '
    idx = 0
    word_sofar = ''
    while index < len(text) and idx < len(word):
        (para_break, break_len) = is_para_break(index, text)
        if para_break:
            if len(word_sofar) > 0:
                assert re.match('^\\s+$', word_sofar), "Found non-empty string at the end of a paragraph that doesn't match any token: |{}|".format(word_sofar)
                word_sofar = ''
            output.write('\n\n')
            index += break_len - 1
        elif re.match('^\\s$', text[index]) and (not re.match('^\\s$', word[idx])):
            word_sofar += text[index]
        else:
            word_sofar += text[index]
            assert text[index].replace('\n', ' ') == word[idx], 'Character mismatch: raw text contains |%s| but the next word is |%s|.' % (word_sofar, word)
            idx += 1
        index += 1
    return (index, word_sofar)

def main(args):
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('plaintext_file', type=str, help='Plaintext file containing the raw input')
    parser.add_argument('conllu_file', type=str, help='CoNLL-U file containing tokens and sentence breaks')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output file name; output to the console if not specified (the default)')
    parser.add_argument('-m', '--mwt_output', default=None, type=str, help='Output file name for MWT expansions; output to the console if not specified (the default)')
    args = parser.parse_args(args=args)
    with open(args.plaintext_file, 'r') as f:
        text = ''.join(f.readlines())
    textlen = len(text)
    if args.output is None:
        output = sys.stdout
    else:
        outdir = os.path.split(args.output)[0]
        os.makedirs(outdir, exist_ok=True)
        output = open(args.output, 'w')
    index = 0
    mwt_expansions = []
    with open(args.conllu_file, 'r') as f:
        buf = ''
        mwtbegin = 0
        mwtend = -1
        expanded = []
        last_comments = ''
        for line in f:
            line = line.strip()
            if len(line):
                if line[0] == '#':
                    if len(last_comments) == 0:
                        last_comments = line
                    continue
                line = line.split('\t')
                if '.' in line[0]:
                    continue
                word = line[1]
                if '-' in line[0]:
                    (mwtbegin, mwtend) = [int(x) for x in line[0].split('-')]
                    lastmwt = word
                    expanded = []
                elif mwtbegin <= int(line[0]) < mwtend:
                    expanded += [word]
                    continue
                elif int(line[0]) == mwtend:
                    expanded += [word]
                    expanded = [x.lower() for x in expanded]
                    mwt_expansions += [(lastmwt, tuple(expanded))]
                    if lastmwt[0].islower() and (not expanded[0][0].islower()):
                        print('Sentence ID with potential wrong MWT expansion: ', last_comments, file=sys.stderr)
                    mwtbegin = 0
                    mwtend = -1
                    lastmwt = None
                    continue
                if len(buf):
                    output.write(buf)
                (index, word_found) = find_next_word(index, text, word, output)
                buf = '0' * (len(word_found) - 1) + ('1' if '-' not in line[0] else '3')
            else:
                if len(buf):
                    assert int(buf[-1]) >= 1
                    output.write(buf[:-1] + '{}'.format(int(buf[-1]) + 1))
                    buf = ''
                last_comments = ''
    output.close()
    mwts = Counter(mwt_expansions)
    if args.mwt_output is None:
        print('MWTs:', mwts)
    else:
        with open(args.mwt_output, 'w') as f:
            json.dump(list(mwts.items()), f)
        print('{} unique MWTs found in data'.format(len(mwts)))
if __name__ == '__main__':
    main(sys.argv[1:])