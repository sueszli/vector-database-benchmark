import re
import argparse
from string import punctuation
import numpy as np
from g2p_en import G2p
import os
ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))

def read_lexicon(lex_path):
    if False:
        print('Hello World!')
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split('\\s+', line.strip('\n'))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text):
    if False:
        for i in range(10):
            print('nop')
    lexicon = read_lexicon(f'{ROOT_DIR}/lexicon/librispeech-lexicon.txt')
    g2p = G2p()
    phones = []
    words = list(filter(lambda x: x not in {'', ' '}, re.split('([,;.\\-\\?\\!\\s+])', text)))
    for w in words:
        if w.lower() in lexicon:
            phones += ['[' + ph + ']' for ph in lexicon[w.lower()]] + ['engsp1']
        else:
            phone = g2p(w)
            if not phone:
                continue
            if phone[0].isalnum():
                phones += ['[' + ph + ']' for ph in phone]
            elif phone == ' ':
                continue
            else:
                phones.pop()
                phones.append('engsp4')
    if 'engsp' in phones[-1]:
        phones.pop()
    mark = '.' if text[-1] != '?' else '?'
    phones = ['<sos/eos>'] + phones + [mark, '<sos/eos>']
    return ' '.join(phones)
if __name__ == '__main__':
    phonemes = preprocess_english('Happy New Year')
    import sys
    from os.path import isfile
    if len(sys.argv) < 2:
        print('Usage: python %s <text>' % sys.argv[0])
        exit()
    text_file = sys.argv[1]
    if isfile(text_file):
        fp = open(text_file, 'r')
        for line in fp:
            phoneme = preprocess_english(line.rstrip())
            print(phoneme)
        fp.close()