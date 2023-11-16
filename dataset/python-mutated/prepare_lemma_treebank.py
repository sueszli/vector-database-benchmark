"""
A script to prepare all lemma datasets.

For example, do
  python -m stanza.utils.datasets.prepare_lemma_treebank TREEBANK
such as
  python -m stanza.utils.datasets.prepare_lemma_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""
import stanza.utils.datasets.common as common
import stanza.utils.datasets.prepare_tokenizer_treebank as prepare_tokenizer_treebank

def check_lemmas(train_file):
    if False:
        i = 10
        return i + 15
    '\n    Check if a treebank has any lemmas in it\n\n    For example, in Vietnamese-VTB, all the words and lemmas are exactly the same\n    in Telugu-MTG, all the lemmas are blank\n    '
    with open(train_file) as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pieces = line.split('\t')
            word = pieces[1].lower().strip()
            lemma = pieces[2].lower().strip()
            if not lemma or lemma == '_' or lemma == '-':
                continue
            if word == lemma:
                continue
            return True
    return False

def process_treebank(treebank, model_type, paths, args):
    if False:
        print('Hello World!')
    if treebank.startswith('UD_'):
        udbase_dir = paths['UDBASE']
        train_conllu = common.find_treebank_dataset_file(treebank, udbase_dir, 'train', 'conllu', fail=True)
        augment = check_lemmas(train_conllu)
        if not augment:
            print('No lemma information found in %s.  Not augmenting the dataset' % train_conllu)
    else:
        augment = True
    prepare_tokenizer_treebank.copy_conllu_treebank(treebank, model_type, paths, paths['LEMMA_DATA_DIR'], augment=augment)

def main():
    if False:
        return 10
    common.main(process_treebank, common.ModelType.LEMMA)
if __name__ == '__main__':
    main()