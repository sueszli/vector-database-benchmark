from collections import namedtuple
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import re
import threading
import numpy as np
import tensorflow as tf
Data = namedtuple('Data', ['X', 'Y', 'MultiYs', 'qid'])

class SampleBuilder:

    def __init__(self, config):
        if False:
            return 10
        self.config = config
        self.kb_raw = self.read_kb()
        self.data_raw = self.read_raw_data()
        self.dict_all = self.gen_dict()
        self.reverse_dict_all = dict(zip(self.dict_all.values(), self.dict_all.keys()))
        tf.logging.info('size of dict: %d' % len(self.dict_all))
        self.kb = self.build_kb()
        self.data_all = self.build_samples()

    def read_kb(self):
        if False:
            while True:
                i = 10
        kb_raw = []
        for line in open(self.config.KB_file):
            (sub, rel, obj) = line.strip().split('|')
            kb_raw.append((sub, rel, obj))
        tf.logging.info('# of KB records: %d' % len(kb_raw))
        return kb_raw

    def read_raw_data(self):
        if False:
            return 10
        data = dict()
        for name in self.config.data_files:
            raw = []
            tf.logging.info('Reading data file {}'.format(self.config.data_files[name]))
            for line in open(self.config.data_files[name]):
                (question, answers) = line.strip().split('\t')
                question = question.replace('],', ']')
                raw.append((question, answers))
            data[name] = raw
        return data

    def build_kb(self):
        if False:
            while True:
                i = 10
        tf.logging.info('Indexing KB...')
        kb = []
        for (sub, rel, obj) in self.kb_raw:
            kb.append([self.dict_all[sub], self.dict_all[rel], self.dict_all[obj]])
        return kb

    def gen_dict(self):
        if False:
            print('Hello World!')
        s = set()
        for (sub, rel, obj) in self.kb_raw:
            s.add(sub)
            s.add(rel)
            s.add(obj)
        for name in self.data_raw:
            for (question, answers) in self.data_raw[name]:
                normal = re.split('\\[[^\\]]+\\]', question)
                for phrase in normal:
                    for word in phrase.split():
                        s.add(word)
        s = list(s)
        d = {s[idx]: idx for idx in range(len(s))}
        return d

    def build_samples(self):
        if False:
            return 10

        def map_entity_idx(text):
            if False:
                print('Hello World!')
            entities = re.findall('\\[[^\\]]+\\]', text)
            for entity in entities:
                entity = entity[1:-1]
                index = self.dict_all[entity]
                text = text.replace('[%s]' % entity, '@%d' % index)
            return text
        data_all = dict()
        for name in self.data_raw:
            (X, Y, MultiYs, qid) = ([], [], [], [])
            for (i, (question, answers)) in enumerate(self.data_raw[name]):
                (qdata, labels) = ([], [])
                question = map_entity_idx(question)
                for word in question.split():
                    if word[0] == '@':
                        qdata.append(int(word[1:]))
                    else:
                        qdata.append(self.dict_all[word])
                for answer in answers.split('|'):
                    labels.append(self.dict_all[answer])
                if len(qdata) > self.config.T_encoder:
                    self.config.T_encoder = len(qdata)
                for label in labels:
                    X.append(qdata)
                    Y.append(label)
                    MultiYs.append(set(labels))
                    qid.append(i)
            data_all[name] = Data(X=X, Y=Y, MultiYs=MultiYs, qid=qid)
        return data_all

def _run_prefetch(prefetch_queue, batch_loader, data, shuffle, one_pass, config):
    if False:
        while True:
            i = 10
    assert len(data.X) == len(data.Y) == len(data.MultiYs) == len(data.qid)
    num_samples = len(data.X)
    batch_size = config.batch_size
    n_sample = 0
    fetch_order = config.rng.permutation(num_samples)
    while True:
        sample_ids = fetch_order[n_sample:n_sample + batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put(batch, block=True)
        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            if one_pass:
                prefetch_queue.put(None, block=True)
            n_sample = 0
            if shuffle:
                fetch_order = config.rng.permutation(num_samples)

class DataReader:

    def __init__(self, config, data, assembler, shuffle=True, one_pass=False, prefetch_num=10):
        if False:
            i = 10
            return i + 15
        self.config = config
        self.data = data
        self.assembler = assembler
        self.batch_loader = BatchLoader(self.config, self.data, self.assembler)
        self.shuffle = shuffle
        self.one_pass = one_pass
        self.prefetch_queue = Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=_run_prefetch, args=(self.prefetch_queue, self.batch_loader, self.data, self.shuffle, self.one_pass, self.config))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self):
        if False:
            while True:
                i = 10
        while True:
            if self.prefetch_queue.empty():
                tf.logging.warning('Waiting for data loading (IO is slow)...')
            batch = self.prefetch_queue.get(block=True)
            if batch is None:
                assert self.one_pass
                tf.logging.info('One pass finished!')
                raise StopIteration()
            yield batch

class BatchLoader:

    def __init__(self, config, data, assembler):
        if False:
            print('Hello World!')
        self.config = config
        self.data = data
        self.assembler = assembler
        self.T_encoder = config.T_encoder
        self.T_decoder = config.T_decoder
        tf.logging.info('T_encoder: %d' % self.T_encoder)
        tf.logging.info('T_decoder: %d' % self.T_decoder)
        tf.logging.info('batch size: %d' % self.config.batch_size)
        self.gt_layout_tokens = config.gt_layout_tokens

    def load_one_batch(self, sample_ids):
        if False:
            while True:
                i = 10
        actual_batch_size = len(sample_ids)
        input_seq_batch = np.zeros((self.T_encoder, actual_batch_size), np.int32)
        seq_len_batch = np.zeros(actual_batch_size, np.int32)
        ans_label_batch = np.zeros(actual_batch_size, np.int32)
        ans_set_labels_list = [None] * actual_batch_size
        question_id_list = [None] * actual_batch_size
        gt_layout_batch = np.zeros((self.T_decoder, actual_batch_size), np.int32)
        for batch_i in range(actual_batch_size):
            idx = sample_ids[batch_i]
            seq_len = len(self.data.X[idx])
            seq_len_batch[batch_i] = seq_len
            input_seq_batch[:seq_len, batch_i] = self.data.X[idx]
            ans_label_batch[batch_i] = self.data.Y[idx]
            ans_set_labels_list[batch_i] = self.data.MultiYs[idx]
            question_id_list[batch_i] = self.data.qid[idx]
            gt_layout_batch[:, batch_i] = self.assembler.module_list2tokens(self.gt_layout_tokens, self.T_decoder)
        batch = dict(input_seq_batch=input_seq_batch, seq_len_batch=seq_len_batch, ans_label_batch=ans_label_batch, gt_layout_batch=gt_layout_batch, ans_set_labels_list=ans_set_labels_list, question_id_list=question_id_list)
        return batch