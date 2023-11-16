import argparse
from collections import defaultdict
import logging
import os
import re
import sys
from stanza.models.common.constant import treebank_to_short_name
from stanza.models.pos.xpos_vocab_utils import DEFAULT_KEY, choose_simplest_factory, XPOSType
from stanza.models.common.doc import *
from stanza.utils.conll import CoNLL
from stanza.utils import default_paths
SHORTNAME_RE = re.compile('[a-z-]+_[a-z0-9]+')
DATA_DIR = default_paths.get_default_paths()['POS_DATA_DIR']
logger = logging.getLogger('stanza')

def get_xpos_factory(shorthand, fn):
    if False:
        i = 10
        return i + 15
    logger.info('Resolving vocab option for {}...'.format(shorthand))
    train_file = os.path.join(DATA_DIR, '{}.train.in.conllu'.format(shorthand))
    if not os.path.exists(train_file):
        raise UserWarning('Training data for {} not found in the data directory, falling back to using WordVocab. To generate the XPOS vocabulary for this treebank properly, please run the following command first:\n\tstanza/utils/datasets/prepare_pos_treebank.py {}'.format(fn, fn))
        key = DEFAULT_KEY
        return key
    doc = CoNLL.conll2doc(input_file=train_file)
    data = doc.get([TEXT, UPOS, XPOS, FEATS], as_sentences=True)
    return choose_simplest_factory(data, shorthand)

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebanks', type=str, default=DATA_DIR, help='Treebanks to process - directory with processed datasets or a file with a list')
    parser.add_argument('--output_file', type=str, default='stanza/models/pos/xpos_vocab_factory.py', help='Where to write the results')
    args = parser.parse_args()
    output_file = args.output_file
    if os.path.isdir(args.treebanks):
        treebanks = os.listdir(args.treebanks)
        treebanks = [x.split('.', maxsplit=1)[0] for x in treebanks]
        treebanks = sorted(set(treebanks))
    elif os.path.exists(args.treebanks):
        with open(args.treebanks) as fin:
            treebanks = sorted(set([x.strip() for x in fin.readlines() if x.strip()]))
    else:
        raise ValueError('Cannot figure out which treebanks to use.   Please set the --treebanks parameter')
    logger.info('Processing the following treebanks: %s' % ' '.join(treebanks))
    shorthands = []
    fullnames = []
    for treebank in treebanks:
        fullnames.append(treebank)
        if SHORTNAME_RE.match(treebank):
            shorthands.append(treebank)
        else:
            shorthands.append(treebank_to_short_name(treebank))
    mapping = defaultdict(list)
    for (sh, fn) in zip(shorthands, fullnames):
        factory = get_xpos_factory(sh, fn)
        mapping[factory].append(sh)
        if sh == 'zh-hans_gsdsimp':
            mapping[factory].append('zh_gsdsimp')
        elif sh == 'no_bokmaal':
            mapping[factory].append('nb_bokmaal')
    mapping[DEFAULT_KEY].append('en_test')
    first = True
    with open(output_file, 'w') as f:
        max_len = max((max((len(x) for x in mapping[key])) for key in mapping))
        print("# This is the XPOS factory method generated automatically from stanza.models.pos.build_xpos_vocab_factory.\n# Please don't edit it!\n\nimport logging\n\nfrom stanza.models.pos.vocab import WordVocab, XPOSVocab\nfrom stanza.models.pos.xpos_vocab_utils import XPOSDescription, XPOSType, build_xpos_vocab, choose_simplest_factory\n\n# using a sublogger makes it easier to test in the unittests\nlogger = logging.getLogger('stanza.models.pos.xpos_vocab_factory')\n\nXPOS_DESCRIPTIONS = {", file=f)
        for (key_idx, key) in enumerate(mapping):
            if key_idx > 0:
                print(file=f)
            for shorthand in sorted(mapping[key]):
                if key.sep is None:
                    sep = 'None'
                else:
                    sep = "'%s'" % key.sep
                print(('    {:%ds}: XPOSDescription({}, {}),' % (max_len + 2)).format("'%s'" % shorthand, key.xpos_type, sep), file=f)
        print('}\n\ndef xpos_vocab_factory(data, shorthand):\n    if shorthand not in XPOS_DESCRIPTIONS:\n        logger.warning("%s is not a known dataset.  Examining the data to choose which xpos vocab to use", shorthand)\n    desc = choose_simplest_factory(data, shorthand)\n    if shorthand in XPOS_DESCRIPTIONS:\n        if XPOS_DESCRIPTIONS[shorthand] != desc:\n            # log instead of throw\n            # otherwise, updating datasets would be unpleasant\n            logger.error("XPOS tagset in %s has apparently changed!  Was %s, is now %s", shorthand, XPOS_DESCRIPTIONS[shorthand], desc)\n    return build_xpos_vocab(desc, data, shorthand)\n', file=f)
    logger.info('Done!')
if __name__ == '__main__':
    main()