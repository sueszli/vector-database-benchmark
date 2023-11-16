import tarfile
from bigdl.dllib.feature.dataset import base
import os
import sys
NEWS20_URL = 'http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz'
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
CLASS_NUM = 20

def download_news20(dest_dir):
    if False:
        while True:
            i = 10
    file_name = '20news-18828.tar.gz'
    file_abs_path = base.maybe_download(file_name, dest_dir, NEWS20_URL)
    with tarfile.open(file_abs_path, 'r:gz') as tar:
        extracted_to = os.path.join(dest_dir, '20news-18828')
        if not os.path.exists(extracted_to):
            print('Extracting %s to %s' % (file_abs_path, extracted_to))
            tar.extractall(dest_dir)
    return extracted_to

def download_glove_w2v(dest_dir):
    if False:
        for i in range(10):
            print('nop')
    file_name = 'glove.6B.zip'
    file_abs_path = base.maybe_download(file_name, dest_dir, GLOVE_URL)
    import zipfile
    with zipfile.ZipFile(file_abs_path, 'r') as zip_ref:
        extracted_to = os.path.join(dest_dir, 'glove.6B')
        if not os.path.exists(extracted_to):
            print('Extracting %s to %s' % (file_abs_path, extracted_to))
            zip_ref.extractall(extracted_to)
    return extracted_to

def get_news20(source_dir='./data/news20/'):
    if False:
        while True:
            i = 10
    '\n    Parse or download news20 if source_dir is empty.\n\n    :param source_dir: The directory storing news data.\n    :return: A list of (tokens, label)\n    '
    news_dir = download_news20(source_dir)
    texts = []
    label_id = 0
    for name in sorted(os.listdir(news_dir)):
        path = os.path.join(news_dir, name)
        label_id += 1
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    kargs = {}
                    if not sys.version_info < (3,):
                        kargs['encoding'] = 'latin-1'
                    with open(fpath, **kargs) as f:
                        content = f.read()
                        texts.append((content, label_id))
    print('Found %s texts.' % len(texts))
    return texts

def get_glove_w2v(source_dir='./data/news20/', dim=100):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse or download the pre-trained glove word2vec if source_dir is empty.\n\n    :param source_dir: The directory storing the pre-trained word2vec\n    :param dim: The dimension of a vector\n    :return: A dict mapping from word to vector\n    '
    w2v_dir = download_glove_w2v(source_dir)
    w2v_path = os.path.join(w2v_dir, 'glove.6B.%sd.txt' % dim)
    kargs = {}
    if not sys.version_info < (3,):
        kargs['encoding'] = 'latin-1'
    with open(w2v_path, **kargs) as w2v_f:
        pre_w2v = {}
        for line in w2v_f.readlines():
            items = line.split(' ')
            pre_w2v[items[0]] = [float(i) for i in items[1:]]
        return pre_w2v
if __name__ == '__main__':
    get_news20('./data/news20/')
    get_glove_w2v('./data/news20/')