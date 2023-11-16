"""Batch reader to seq2seq attention model, with bucketing support."""
from collections import namedtuple
from random import shuffle
from threading import Thread
import time
import numpy as np
import six
from six.moves import queue as Queue
from six.moves import xrange
import tensorflow as tf
import data
ModelInput = namedtuple('ModelInput', 'enc_input dec_input target enc_len dec_len origin_article origin_abstract')
BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100

class Batcher(object):
    """Batch reader with shuffling and bucketing support."""

    def __init__(self, data_path, vocab, hps, article_key, abstract_key, max_article_sentences, max_abstract_sentences, bucketing=True, truncate_input=False):
        if False:
            print('Hello World!')
        'Batcher constructor.\n\n    Args:\n      data_path: tf.Example filepattern.\n      vocab: Vocabulary.\n      hps: Seq2SeqAttention model hyperparameters.\n      article_key: article feature key in tf.Example.\n      abstract_key: abstract feature key in tf.Example.\n      max_article_sentences: Max number of sentences used from article.\n      max_abstract_sentences: Max number of sentences used from abstract.\n      bucketing: Whether bucket articles of similar length into the same batch.\n      truncate_input: Whether to truncate input that is too long. Alternative is\n        to discard such examples.\n    '
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._article_key = article_key
        self._abstract_key = abstract_key
        self._max_article_sentences = max_article_sentences
        self._max_abstract_sentences = max_abstract_sentences
        self._bucketing = bucketing
        self._truncate_input = truncate_input
        self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
        self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)
        self._input_threads = []
        for _ in xrange(16):
            self._input_threads.append(Thread(target=self._FillInputQueue))
            self._input_threads[-1].daemon = True
            self._input_threads[-1].start()
        self._bucketing_threads = []
        for _ in xrange(4):
            self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
            self._bucketing_threads[-1].daemon = True
            self._bucketing_threads[-1].start()
        self._watch_thread = Thread(target=self._WatchThreads)
        self._watch_thread.daemon = True
        self._watch_thread.start()

    def NextBatch(self):
        if False:
            while True:
                i = 10
        'Returns a batch of inputs for seq2seq attention model.\n\n    Returns:\n      enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].\n      dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].\n      target_batch: A batch of targets [batch_size, hps.dec_timestamps].\n      enc_input_len: encoder input lengths of the batch.\n      dec_input_len: decoder input lengths of the batch.\n      loss_weights: weights for loss function, 1 if not padded, 0 if padded.\n      origin_articles: original article words.\n      origin_abstracts: original abstract words.\n    '
        enc_batch = np.zeros((self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
        enc_input_lens = np.zeros(self._hps.batch_size, dtype=np.int32)
        dec_batch = np.zeros((self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
        dec_output_lens = np.zeros(self._hps.batch_size, dtype=np.int32)
        target_batch = np.zeros((self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
        loss_weights = np.zeros((self._hps.batch_size, self._hps.dec_timesteps), dtype=np.float32)
        origin_articles = ['None'] * self._hps.batch_size
        origin_abstracts = ['None'] * self._hps.batch_size
        buckets = self._bucket_input_queue.get()
        for i in xrange(self._hps.batch_size):
            (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len, article, abstract) = buckets[i]
            origin_articles[i] = article
            origin_abstracts[i] = abstract
            enc_input_lens[i] = enc_input_len
            dec_output_lens[i] = dec_output_len
            enc_batch[i, :] = enc_inputs[:]
            dec_batch[i, :] = dec_inputs[:]
            target_batch[i, :] = targets[:]
            for j in xrange(dec_output_len):
                loss_weights[i][j] = 1
        return (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens, loss_weights, origin_articles, origin_abstracts)

    def _FillInputQueue(self):
        if False:
            i = 10
            return i + 15
        'Fill input queue with ModelInput.'
        start_id = self._vocab.WordToId(data.SENTENCE_START)
        end_id = self._vocab.WordToId(data.SENTENCE_END)
        pad_id = self._vocab.WordToId(data.PAD_TOKEN)
        input_gen = self._TextGenerator(data.ExampleGen(self._data_path))
        while True:
            (article, abstract) = six.next(input_gen)
            article_sentences = [sent.strip() for sent in data.ToSentences(article, include_token=False)]
            abstract_sentences = [sent.strip() for sent in data.ToSentences(abstract, include_token=False)]
            enc_inputs = []
            dec_inputs = [start_id]
            for i in xrange(min(self._max_article_sentences, len(article_sentences))):
                enc_inputs += data.GetWordIds(article_sentences[i], self._vocab)
            for i in xrange(min(self._max_abstract_sentences, len(abstract_sentences))):
                dec_inputs += data.GetWordIds(abstract_sentences[i], self._vocab)
            if len(enc_inputs) < self._hps.min_input_len or len(dec_inputs) < self._hps.min_input_len:
                tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d', len(enc_inputs), len(dec_inputs))
                continue
            if not self._truncate_input:
                if len(enc_inputs) > self._hps.enc_timesteps or len(dec_inputs) > self._hps.dec_timesteps:
                    tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d', len(enc_inputs), len(dec_inputs))
                    continue
            else:
                if len(enc_inputs) > self._hps.enc_timesteps:
                    enc_inputs = enc_inputs[:self._hps.enc_timesteps]
                if len(dec_inputs) > self._hps.dec_timesteps:
                    dec_inputs = dec_inputs[:self._hps.dec_timesteps]
            targets = dec_inputs[1:]
            targets.append(end_id)
            enc_input_len = len(enc_inputs)
            dec_output_len = len(targets)
            while len(enc_inputs) < self._hps.enc_timesteps:
                enc_inputs.append(pad_id)
            while len(dec_inputs) < self._hps.dec_timesteps:
                dec_inputs.append(end_id)
            while len(targets) < self._hps.dec_timesteps:
                targets.append(end_id)
            element = ModelInput(enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len, ' '.join(article_sentences), ' '.join(abstract_sentences))
            self._input_queue.put(element)

    def _FillBucketInputQueue(self):
        if False:
            i = 10
            return i + 15
        'Fill bucketed batches into the bucket_input_queue.'
        while True:
            inputs = []
            for _ in xrange(self._hps.batch_size * BUCKET_CACHE_BATCH):
                inputs.append(self._input_queue.get())
            if self._bucketing:
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)
            batches = []
            for i in xrange(0, len(inputs), self._hps.batch_size):
                batches.append(inputs[i:i + self._hps.batch_size])
            shuffle(batches)
            for b in batches:
                self._bucket_input_queue.put(b)

    def _WatchThreads(self):
        if False:
            return 10
        'Watch the daemon input threads and restart if dead.'
        while True:
            time.sleep(60)
            input_threads = []
            for t in self._input_threads:
                if t.is_alive():
                    input_threads.append(t)
                else:
                    tf.logging.error('Found input thread dead.')
                    new_t = Thread(target=self._FillInputQueue)
                    input_threads.append(new_t)
                    input_threads[-1].daemon = True
                    input_threads[-1].start()
            self._input_threads = input_threads
            bucketing_threads = []
            for t in self._bucketing_threads:
                if t.is_alive():
                    bucketing_threads.append(t)
                else:
                    tf.logging.error('Found bucketing thread dead.')
                    new_t = Thread(target=self._FillBucketInputQueue)
                    bucketing_threads.append(new_t)
                    bucketing_threads[-1].daemon = True
                    bucketing_threads[-1].start()
            self._bucketing_threads = bucketing_threads

    def _TextGenerator(self, example_gen):
        if False:
            i = 10
            return i + 15
        'Generates article and abstract text from tf.Example.'
        while True:
            e = six.next(example_gen)
            try:
                article_text = self._GetExFeatureText(e, self._article_key)
                abstract_text = self._GetExFeatureText(e, self._abstract_key)
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
            yield (article_text, abstract_text)

    def _GetExFeatureText(self, ex, key):
        if False:
            while True:
                i = 10
        'Extract text for a feature from td.Example.\n\n    Args:\n      ex: tf.Example.\n      key: key of the feature to be extracted.\n    Returns:\n      feature: a feature text extracted.\n    '
        return ex.features.feature[key].bytes_list.value[0]