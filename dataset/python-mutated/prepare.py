from future import standard_library
standard_library.install_aliases()
import h5py
from collections import defaultdict
import numpy as np
import os
from neon import logger as neon_logger
from neon.data.text_preprocessing import clean_string
from neon.util.compat import pickle

def build_data_train(path='.', filepath='labeledTrainData.tsv', vocab_file=None, vocab=None, skip_headers=True, train_ratio=0.8):
    if False:
        return 10
    '\n    Loads the data file and spits out a h5 file with record of\n    {y, review_text, review_int}\n    Typically two passes over the data.\n    1st pass is for vocab and pre-processing. (WARNING: to get phrases, we need to go\n    though multiple passes). 2nd pass is converting text into integers. We will deal with integers\n    from thereafter.\n\n    WARNING: we use h5 just as proof of concept for handling large datasets\n    Datasets may fit entirely in memory as numpy as array\n\n    '
    fname_h5 = filepath + '.h5'
    if vocab_file is None:
        fname_vocab = filepath + '.vocab'
    else:
        fname_vocab = vocab_file
    if not os.path.exists(fname_h5) or not os.path.exists(fname_vocab):
        h5f = h5py.File(fname_h5, 'w')
        (shape, maxshape) = ((2 ** 16,), (None,))
        dt = np.dtype([('y', np.uint8), ('split', np.bool), ('num_words', np.uint16), ('text', h5py.special_dtype(vlen=str))])
        reviews_text = h5f.create_dataset('reviews', shape=shape, maxshape=maxshape, dtype=dt, compression='gzip')
        reviews_train = h5f.create_dataset('train', shape=shape, maxshape=maxshape, dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')
        reviews_valid = h5f.create_dataset('valid', shape=shape, maxshape=maxshape, dtype=h5py.special_dtype(vlen=np.int32), compression='gzip')
        wdata = np.zeros((1,), dtype=dt)
        build_vocab = False
        if vocab is None:
            vocab = defaultdict(int)
            build_vocab = True
        nsamples = 0
        f = open(filepath, 'r')
        if skip_headers:
            f.readline()
        for (i, line) in enumerate(f):
            (_, rating, review) = line.strip().split('\t')
            review = clean_string(review)
            review_words = review.strip().split()
            num_words = len(review_words)
            split = int(np.random.rand() < train_ratio)
            wdata['y'] = int(float(rating))
            wdata['text'] = review
            wdata['num_words'] = num_words
            wdata['split'] = split
            reviews_text[i] = wdata
            if build_vocab:
                for word in review_words:
                    vocab[word] += 1
            nsamples += 1
        (ratings, counts) = np.unique(reviews_text['y'][:nsamples], return_counts=True)
        (sen_len, sen_len_counts) = np.unique(reviews_text['num_words'][:nsamples], return_counts=True)
        vocab_size = len(vocab)
        nclass = len(ratings)
        reviews_text.attrs['vocab_size'] = vocab_size
        reviews_text.attrs['nrows'] = nsamples
        reviews_text.attrs['nclass'] = nclass
        reviews_text.attrs['class_distribution'] = counts
        neon_logger.display('vocabulary size - {}'.format(vocab_size))
        neon_logger.display('# of samples - {}'.format(nsamples))
        neon_logger.display('# of classes {}'.format(nclass))
        neon_logger.display('class distribution - {} {}'.format(ratings, counts))
        sen_counts = list(zip(sen_len, sen_len_counts))
        sen_counts = sorted(sen_counts, key=lambda kv: kv[1], reverse=True)
        neon_logger.display('sentence length - {} {} {}'.format(len(sen_len), sen_len, sen_len_counts))
        if build_vocab:
            vocab_sorted = sorted(list(vocab.items()), key=lambda kv: kv[1], reverse=True)
            vocab = {}
            for (i, t) in enumerate(list(zip(*vocab_sorted))[0]):
                vocab[t] = i
        ntrain = 0
        nvalid = 0
        for i in range(nsamples):
            text = reviews_text[i]['text']
            y = int(reviews_text[i]['y'])
            split = reviews_text[i]['split']
            text_int = [y] + [vocab[t] for t in text.strip().split()]
            if split:
                reviews_train[ntrain] = text_int
                ntrain += 1
            else:
                reviews_valid[nvalid] = text_int
                nvalid += 1
        reviews_text.attrs['ntrain'] = ntrain
        reviews_text.attrs['nvalid'] = nvalid
        neon_logger.display('# of train - {0}, # of valid - {1}'.format(reviews_text.attrs['ntrain'], reviews_text.attrs['nvalid']))
        h5f.close()
        f.close()
    if not os.path.exists(fname_vocab):
        rev_vocab = {}
        for (wrd, wrd_id) in vocab.items():
            rev_vocab[wrd_id] = wrd
        neon_logger.display('vocabulary from IMDB dataset is saved into {}'.format(fname_vocab))
        pickle.dump((vocab, rev_vocab), open(fname_vocab, 'wb'), 2)
    return (fname_h5, fname_vocab)