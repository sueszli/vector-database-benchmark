__author__ = 'nastra'
import logging, gensim
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def build_and_save_wikipedia_model(save_path):
    if False:
        print('Hello World!')
    id2word = gensim.corpora.Dictionary.load_from_text('data/wiki_en_wordids.txt')
    corpus = gensim.corpora.MmCorpus('data/wiki_en_output_tfidf.mm')
    mm = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
    mm.save(save_path)
    return mm

def load_wikipedia_model(path):
    if False:
        for i in range(10):
            print('nop')
    try:
        return gensim.models.LdaModel.load(path)
    except:
        return build_and_save_wikipedia_model(path)
model_path = 'data/wikipedia_lda.pk1'
model = load_wikipedia_model(model_path)
topics = []
for doc in model:
    topics.append(model[doc])
lens = np.array([len(t) for t in topics])
print(np.mean(lens <= 10))
print(np.mean(lens))
counts = np.zeros(100)
for doc_topic in topics:
    for (index, _) in doc_topic:
        counts[index] += 1
words = model.show_topic(counts.argmax(), 20)
print(words)
print(120 * '=')
words = model.show_topic(counts.argmin(), 20)
print(words)
print(120 * '=')