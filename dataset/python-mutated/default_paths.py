import os

def get_default_paths():
    if False:
        print('Hello World!')
    '\n    Gets base paths for the data directories\n\n    If DATA_ROOT is set in the environment, use that as the root\n    otherwise use "./data"\n    individual paths can also be set in the environment\n    '
    DATA_ROOT = os.environ.get('DATA_ROOT', 'data')
    defaults = {'TOKENIZE_DATA_DIR': DATA_ROOT + '/tokenize', 'MWT_DATA_DIR': DATA_ROOT + '/mwt', 'LEMMA_DATA_DIR': DATA_ROOT + '/lemma', 'POS_DATA_DIR': DATA_ROOT + '/pos', 'DEPPARSE_DATA_DIR': DATA_ROOT + '/depparse', 'ETE_DATA_DIR': DATA_ROOT + '/ete', 'NER_DATA_DIR': DATA_ROOT + '/ner', 'CHARLM_DATA_DIR': DATA_ROOT + '/charlm', 'SENTIMENT_DATA_DIR': DATA_ROOT + '/sentiment', 'CONSTITUENCY_DATA_DIR': DATA_ROOT + '/constituency', 'WORDVEC_DIR': 'extern_data/wordvec', 'UDBASE': 'extern_data/ud2/ud-treebanks-v2.11', 'UDBASE_GIT': 'extern_data/ud2/git', 'NERBASE': 'extern_data/ner', 'CONSTITUENCY_BASE': 'extern_data/constituency', 'SENTIMENT_BASE': 'extern_data/sentiment', 'HANDPARSED_DIR': 'extern_data/handparsed-treebank', 'BIO_UD_DIR': 'extern_data/bio', 'STANZA_EXTERN_DIR': 'extern_data'}
    paths = {'DATA_ROOT': DATA_ROOT}
    for (k, v) in defaults.items():
        paths[k] = os.environ.get(k, v)
    return paths