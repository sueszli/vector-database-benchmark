"""
Corpus for the DML-CZ project.
"""
import logging
import os.path
from gensim import interfaces, matutils
import dictionary
logger = logging.getLogger('gensim.corpora.dmlcorpus')

class DmlConfig:
    """
    DmlConfig contains parameters necessary for the abstraction of a 'corpus of
    articles' (see the `DmlCorpus` class).

    Articles may come from different sources (=different locations on disk/network,
    different file formats etc.), so the main purpose of DmlConfig is to keep all
    sources in one place.

    Apart from glueing sources together, DmlConfig also decides where to store
    output files and which articles to accept for the corpus (= an additional filter
    over the sources).
    """

    def __init__(self, configId, resultDir, acceptLangs=None):
        if False:
            i = 10
            return i + 15
        self.resultDir = resultDir
        self.configId = configId
        self.sources = {}
        if acceptLangs is None:
            acceptLangs = {'any'}
        self.acceptLangs = set(acceptLangs)
        logger.info('initialized %s', self)

    def resultFile(self, fname):
        if False:
            while True:
                i = 10
        return os.path.join(self.resultDir, self.configId + '_' + fname)

    def acceptArticle(self, metadata):
        if False:
            while True:
                i = 10
        lang = metadata.get('language', 'unk')
        if 'any' not in self.acceptLangs and lang not in self.acceptLangs:
            return False
        return True

    def addSource(self, source):
        if False:
            i = 10
            return i + 15
        sourceId = str(source)
        assert sourceId not in self.sources, 'source %s already present in the config!' % sourceId
        self.sources[sourceId] = source

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s<id=%s, sources=[%s], acceptLangs=[%s]>' % (self.__class__.__name__, self.configId, ', '.join(self.sources.iterkeys()), ', '.join(self.acceptLangs))

class DmlCorpus(interfaces.CorpusABC):
    """
    DmlCorpus implements a collection of articles. It is initialized via a DmlConfig
    object, which holds information about where to look for the articles and how
    to process them.

    Apart from being a regular corpus (bag-of-words iterable with a `len()` method),
    DmlCorpus has methods for building a dictionary (mapping between words and
    their ids).
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.documents = []
        self.config = None
        self.dictionary = dictionary.Dictionary()

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.documents)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        '\n        The function that defines a corpus -- iterating over the corpus yields\n        bag-of-words vectors, one for each document.\n\n        A bag-of-words vector is simply a list of ``(tokenId, tokenCount)`` 2-tuples.\n        '
        for (docNo, (sourceId, docUri)) in enumerate(self.documents):
            source = self.config.sources[sourceId]
            contents = source.getContent(docUri)
            words = [source.normalizeWord(word) for word in source.tokenize(contents)]
            yield self.dictionary.doc2bow(words, allowUpdate=False)

    def buildDictionary(self):
        if False:
            while True:
                i = 10
        '\n        Populate dictionary mapping and statistics.\n\n        This is done by sequentially retrieving the article fulltexts, splitting\n        them into tokens and converting tokens to their ids (creating new ids as\n        necessary).\n        '
        logger.info('creating dictionary from %i articles', len(self.documents))
        self.dictionary = dictionary.Dictionary()
        numPositions = 0
        for (docNo, (sourceId, docUri)) in enumerate(self.documents):
            if docNo % 1000 == 0:
                logger.info('PROGRESS: at document #%i/%i (%s, %s)', docNo, len(self.documents), sourceId, docUri)
            source = self.config.sources[sourceId]
            contents = source.getContent(docUri)
            words = [source.normalizeWord(word) for word in source.tokenize(contents)]
            numPositions += len(words)
            _ = self.dictionary.doc2bow(words, allowUpdate=True)
        logger.info('built %s from %i documents (total %i corpus positions)', self.dictionary, len(self.documents), numPositions)

    def processConfig(self, config, shuffle=False):
        if False:
            return 10
        "\n        Parse the directories specified in the config, looking for suitable articles.\n\n        This updates the self.documents var, which keeps a list of (source id,\n        article uri) 2-tuples. Each tuple is a unique identifier of one article.\n\n        Note that some articles are ignored based on config settings (for example\n        if the article's language doesn't match any language specified in the\n        config etc.).\n        "
        self.config = config
        self.documents = []
        logger.info('processing config %s', config)
        for (sourceId, source) in config.sources.iteritems():
            logger.info("processing source '%s'", sourceId)
            accepted = []
            for articleUri in source.findArticles():
                meta = source.getMeta(articleUri)
                if config.acceptArticle(meta):
                    accepted.append((sourceId, articleUri))
            logger.info("accepted %i articles for source '%s'", len(accepted), sourceId)
            self.documents.extend(accepted)
        if not self.documents:
            logger.warning('no articles at all found from the config; something went wrong!')
        if shuffle:
            logger.info('shuffling %i documents for random order', len(self.documents))
            import random
            random.shuffle(self.documents)
        logger.info('accepted total of %i articles for %s', len(self.documents), str(config))

    def saveDictionary(self, fname):
        if False:
            return 10
        logger.info('saving dictionary mapping to %s', fname)
        fout = open(fname, 'w')
        for (tokenId, token) in self.dictionary.id2token.iteritems():
            fout.write('%i\t%s\n' % (tokenId, token))
        fout.close()

    @staticmethod
    def loadDictionary(fname):
        if False:
            print('Hello World!')
        result = {}
        for (lineNo, line) in enumerate(open(fname)):
            pair = line[:-1].split('\t')
            if len(pair) != 2:
                continue
            (wordId, word) = pair
            result[int(wordId)] = word
        return result

    def saveDocuments(self, fname):
        if False:
            while True:
                i = 10
        logger.info('saving documents mapping to %s', fname)
        fout = open(fname, 'w')
        for (docNo, docId) in enumerate(self.documents):
            (sourceId, docUri) = docId
            (intId, pathId) = docUri
            fout.write('%i\t%s\n' % (docNo, repr(docId)))
        fout.close()

    def saveAsText(self):
        if False:
            print('Hello World!')
        '\n        Store the corpus to disk, in a human-readable text format.\n\n        This actually saves multiple files:\n\n        1. Pure document-term co-occurence frequency counts, as a Matrix Market file.\n        2. Token to integer mapping, as a text file.\n        3. Document to document URI mapping, as a text file.\n\n        The exact filesystem paths and filenames are determined from the config.\n        '
        self.saveDictionary(self.config.resultFile('wordids.txt'))
        self.saveDocuments(self.config.resultFile('docids.txt'))
        matutils.MmWriter.writeCorpus(self.config.resultFile('bow.mm'), self)

    def articleDir(self, docNo):
        if False:
            print('Hello World!')
        '\n        Return absolute normalized path on filesystem to article no. `docNo`.\n        '
        (sourceId, (_, outPath)) = self.documents[docNo]
        source = self.config.sources[sourceId]
        return os.path.join(source.baseDir, outPath)

    def getMeta(self, docNo):
        if False:
            return 10
        '\n        Return metadata for article no. `docNo`.\n        '
        (sourceId, uri) = self.documents[docNo]
        source = self.config.sources[sourceId]
        return source.getMeta(uri)