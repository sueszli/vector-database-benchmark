"""
USAGE: %(program)s LANGUAGE METHOD
    Generate similar.xml files, using a previously built model for METHOD.

Example: ./gensim_xml.py eng lsi
"""
import logging
import sys
import os.path
from gensim.corpora import dmlcorpus, MmCorpus
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity
import gensim_build
DRY_RUN = False
MIN_SCORE = 0.0
MAX_SIMILAR = 10
SAVE_EMPTY = True
ARTICLE = '\n    <article weight="%(score)f">\n        <authors>\n            <author>%(author)s</author>\n        </authors>\n        <title>%(title)s</title>\n        <suffix>%(suffix)s</suffix>\n        <links>\n            <link source="%(source)s" id="%(intId)s" path="%(pathId)s"/>\n        </links>\n    </article>'
SIMILAR = '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n<related>%s\n</related>\n'

def generateSimilar(corpus, index, method):
    if False:
        while True:
            i = 10
    for (docNo, topSims) in enumerate(index):
        outfile = os.path.join(corpus.articleDir(docNo), 'similar_%s.xml' % method)
        articles = []
        for (docNo2, score) in topSims:
            if score > MIN_SCORE and docNo != docNo2:
                (source, (intId, pathId)) = corpus.documents[docNo2]
                meta = corpus.getMeta(docNo2)
                (suffix, author, title) = ('', meta.get('author', ''), meta.get('title', ''))
                articles.append(ARTICLE % locals())
                if len(articles) >= MAX_SIMILAR:
                    break
        if SAVE_EMPTY or articles:
            output = ''.join(articles)
            if not DRY_RUN:
                logging.info('generating %s (%i similars)', outfile, len(articles))
                outfile = open(outfile, 'w')
                outfile.write(SIMILAR % output)
                outfile.close()
            else:
                logging.info('would be generating %s (%i similars):%s\n', outfile, len(articles), output)
        else:
            logging.debug('skipping %s (no similar found)', outfile)
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info('running %s', ' '.join(sys.argv))
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    language = sys.argv[1]
    method = sys.argv[2].strip().lower()
    logging.info('loading corpus mappings')
    config = dmlcorpus.DmlConfig('%s_%s' % (gensim_build.PREFIX, language), resultDir=gensim_build.RESULT_DIR, acceptLangs=[language])
    logging.info('loading word id mapping from %s', config.resultFile('wordids.txt'))
    id2word = dmlcorpus.DmlCorpus.loadDictionary(config.resultFile('wordids.txt'))
    logging.info('loaded %i word ids', len(id2word))
    corpus = dmlcorpus.DmlCorpus.load(config.resultFile('.pkl'))
    input = MmCorpus(config.resultFile('_%s.mm' % method))
    assert len(input) == len(corpus), 'corpus size mismatch (%i vs %i): run ./gensim_genmodel.py again' % (len(input), len(corpus))
    if method == 'lsi' or method == 'rp':
        index = MatrixSimilarity(input, num_best=MAX_SIMILAR + 1, num_features=input.numTerms)
    else:
        index = SparseMatrixSimilarity(input, num_best=MAX_SIMILAR + 1)
    index.normalize = False
    generateSimilar(corpus, index, method)
    logging.info('finished running %s', program)