import tensorflow as tf
import numpy as np
import random
from recommenders.models.deeprec.io.iterator import BaseIterator
from recommenders.models.deeprec.deeprec_utils import load_dict
__all__ = ['SequentialIterator']

class SequentialIterator(BaseIterator):

    def __init__(self, hparams, graph, col_spliter='\t'):
        if False:
            for i in range(10):
                print('nop')
        'Initialize an iterator. Create necessary placeholders for the model.\n\n        Args:\n            hparams (object): Global hyper-parameters. Some key settings such as #_feature and #_field are there.\n            graph (object): The running graph. All created placeholder will be added to this graph.\n            col_spliter (str): Column splitter in one line.\n        '
        self.col_spliter = col_spliter
        (user_vocab, item_vocab, cate_vocab) = (hparams.user_vocab, hparams.item_vocab, hparams.cate_vocab)
        (self.userdict, self.itemdict, self.catedict) = (load_dict(user_vocab), load_dict(item_vocab), load_dict(cate_vocab))
        self.max_seq_length = hparams.max_seq_length
        self.batch_size = hparams.batch_size
        self.iter_data = dict()
        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name='label')
            self.users = tf.compat.v1.placeholder(tf.int32, [None], name='users')
            self.items = tf.compat.v1.placeholder(tf.int32, [None], name='items')
            self.cates = tf.compat.v1.placeholder(tf.int32, [None], name='cates')
            self.item_history = tf.compat.v1.placeholder(tf.int32, [None, self.max_seq_length], name='item_history')
            self.item_cate_history = tf.compat.v1.placeholder(tf.int32, [None, self.max_seq_length], name='item_cate_history')
            self.mask = tf.compat.v1.placeholder(tf.int32, [None, self.max_seq_length], name='mask')
            self.time = tf.compat.v1.placeholder(tf.float32, [None], name='time')
            self.time_diff = tf.compat.v1.placeholder(tf.float32, [None, self.max_seq_length], name='time_diff')
            self.time_from_first_action = tf.compat.v1.placeholder(tf.float32, [None, self.max_seq_length], name='time_from_first_action')
            self.time_to_now = tf.compat.v1.placeholder(tf.float32, [None, self.max_seq_length], name='time_to_now')

    def parse_file(self, input_file):
        if False:
            return 10
        'Parse the file to A list ready to be used for downstream tasks.\n\n        Args:\n            input_file: One of train, valid or test file which has never been parsed.\n\n        Returns:\n            list: A list with parsing result.\n        '
        with open(input_file, 'r') as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def parser_one_line(self, line):
        if False:
            while True:
                i = 10
        'Parse one string line into feature values.\n\n        Args:\n            line (str): a string indicating one instance.\n                This string contains tab-separated values including:\n                label, user_hash, item_hash, item_cate, operation_time, item_history_sequence,\n                item_cate_history_sequence, and time_history_sequence.\n\n        Returns:\n            list: Parsed results including `label`, `user_id`, `item_id`, `item_cate`, `item_history_sequence`, `cate_history_sequence`,\n            `current_time`, `time_diff`, `time_from_first_action`, `time_to_now`.\n\n        '
        words = line.strip().split(self.col_spliter)
        label = int(words[0])
        user_id = self.userdict[words[1]] if words[1] in self.userdict else 0
        item_id = self.itemdict[words[2]] if words[2] in self.itemdict else 0
        item_cate = self.catedict[words[3]] if words[3] in self.catedict else 0
        current_time = float(words[4])
        item_history_sequence = []
        cate_history_sequence = []
        time_history_sequence = []
        item_history_words = words[5].strip().split(',')
        for item in item_history_words:
            item_history_sequence.append(self.itemdict[item] if item in self.itemdict else 0)
        cate_history_words = words[6].strip().split(',')
        for cate in cate_history_words:
            cate_history_sequence.append(self.catedict[cate] if cate in self.catedict else 0)
        time_history_words = words[7].strip().split(',')
        time_history_sequence = [float(i) for i in time_history_words]
        time_range = 3600 * 24
        time_diff = []
        for i in range(len(time_history_sequence) - 1):
            diff = (time_history_sequence[i + 1] - time_history_sequence[i]) / time_range
            diff = max(diff, 0.5)
            time_diff.append(diff)
        last_diff = (current_time - time_history_sequence[-1]) / time_range
        last_diff = max(last_diff, 0.5)
        time_diff.append(last_diff)
        time_diff = np.log(time_diff)
        time_from_first_action = []
        first_time = time_history_sequence[0]
        time_from_first_action = [(t - first_time) / time_range for t in time_history_sequence[1:]]
        time_from_first_action = [max(t, 0.5) for t in time_from_first_action]
        last_diff = (current_time - first_time) / time_range
        last_diff = max(last_diff, 0.5)
        time_from_first_action.append(last_diff)
        time_from_first_action = np.log(time_from_first_action)
        time_to_now = []
        time_to_now = [(current_time - t) / time_range for t in time_history_sequence]
        time_to_now = [max(t, 0.5) for t in time_to_now]
        time_to_now = np.log(time_to_now)
        return (label, user_id, item_id, item_cate, item_history_sequence, cate_history_sequence, current_time, time_diff, time_from_first_action, time_to_now)

    def load_data_from_file(self, infile, batch_num_ngs=0, min_seq_length=1):
        if False:
            for i in range(10):
                print('nop')
        'Read and parse data from a file.\n\n        Args:\n            infile (str): Text input file. Each line in this file is an instance.\n            batch_num_ngs (int): The number of negative sampling here in batch.\n                0 represents that there is no need to do negative sampling here.\n            min_seq_length (int): The minimum number of a sequence length.\n                Sequences with length lower than min_seq_length will be ignored.\n\n        Yields:\n            object: An iterator that yields parsed results, in the format of graph `feed_dict`.\n        '
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_history_batch = []
        item_cate_history_batch = []
        time_list = []
        time_diff_list = []
        time_from_first_action_list = []
        time_to_now_list = []
        cnt = 0
        if infile not in self.iter_data:
            lines = self.parse_file(infile)
            self.iter_data[infile] = lines
        else:
            lines = self.iter_data[infile]
        if batch_num_ngs > 0:
            random.shuffle(lines)
        for line in lines:
            if not line:
                continue
            (label, user_id, item_id, item_cate, item_history_sequence, item_cate_history_sequence, current_time, time_diff, time_from_first_action, time_to_now) = line
            if len(item_history_sequence) < min_seq_length:
                continue
            label_list.append(label)
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_history_batch.append(item_history_sequence)
            item_cate_history_batch.append(item_cate_history_sequence)
            time_list.append(current_time)
            time_diff_list.append(time_diff)
            time_from_first_action_list.append(time_from_first_action)
            time_to_now_list.append(time_to_now)
            cnt += 1
            if cnt == self.batch_size:
                res = self._convert_data(label_list, user_list, item_list, item_cate_list, item_history_batch, item_cate_history_batch, time_list, time_diff_list, time_from_first_action_list, time_to_now_list, batch_num_ngs)
                batch_input = self.gen_feed_dict(res)
                yield (batch_input if batch_input else None)
                label_list = []
                user_list = []
                item_list = []
                item_cate_list = []
                item_history_batch = []
                item_cate_history_batch = []
                time_list = []
                time_diff_list = []
                time_from_first_action_list = []
                time_to_now_list = []
                cnt = 0
        if cnt > 0:
            res = self._convert_data(label_list, user_list, item_list, item_cate_list, item_history_batch, item_cate_history_batch, time_list, time_diff_list, time_from_first_action_list, time_to_now_list, batch_num_ngs)
            batch_input = self.gen_feed_dict(res)
            yield (batch_input if batch_input else None)

    def _convert_data(self, label_list, user_list, item_list, item_cate_list, item_history_batch, item_cate_history_batch, time_list, time_diff_list, time_from_first_action_list, time_to_now_list, batch_num_ngs):
        if False:
            print('Hello World!')
        'Convert data into numpy arrays that are good for further model operation.\n\n        Args:\n            label_list (list): A list of ground-truth labels.\n            user_list (list): A list of user indexes.\n            item_list (list): A list of item indexes.\n            item_cate_list (list): A list of category indexes.\n            item_history_batch (list): A list of item history indexes.\n            item_cate_history_batch (list): A list of category history indexes.\n            time_list (list): A list of current timestamp.\n            time_diff_list (list): A list of timestamp between each sequential operations.\n            time_from_first_action_list (list): A list of timestamp from the first operation.\n            time_to_now_list (list): A list of timestamp to the current time.\n            batch_num_ngs (int): The number of negative sampling while training in mini-batch.\n\n        Returns:\n            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.\n        '
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return
            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            user_list_all = np.asarray([[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32).flatten()
            time_list_all = np.asarray([[t] * (batch_num_ngs + 1) for t in time_list], dtype=np.float32).flatten()
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros((instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)).astype('int32')
            item_cate_history_batch_all = np.zeros((instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)).astype('int32')
            time_diff_batch = np.zeros((instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)).astype('float32')
            time_from_first_action_batch = np.zeros((instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)).astype('float32')
            time_to_now_batch = np.zeros((instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)).astype('float32')
            mask = np.zeros((instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)).astype('float32')
            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                for index in range(batch_num_ngs + 1):
                    item_history_batch_all[i * (batch_num_ngs + 1) + index, :this_length] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32)
                    item_cate_history_batch_all[i * (batch_num_ngs + 1) + index, :this_length] = np.asarray(item_cate_history_batch[i][-this_length:], dtype=np.int32)
                    mask[i * (batch_num_ngs + 1) + index, :this_length] = 1.0
                    time_diff_batch[i * (batch_num_ngs + 1) + index, :this_length] = np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
                    time_from_first_action_batch[i * (batch_num_ngs + 1) + index, :this_length] = np.asarray(time_from_first_action_list[i][-this_length:], dtype=np.float32)
                    time_to_now_batch[i * (batch_num_ngs + 1) + index, :this_length] = np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break
            res = {}
            res['labels'] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
            res['users'] = user_list_all
            res['items'] = np.asarray(item_list_all, dtype=np.int32)
            res['cates'] = np.asarray(item_cate_list_all, dtype=np.int32)
            res['item_history'] = item_history_batch_all
            res['item_cate_history'] = item_cate_history_batch_all
            res['mask'] = mask
            res['time'] = time_list_all
            res['time_diff'] = time_diff_batch
            res['time_from_first_action'] = time_from_first_action_batch
            res['time_to_now'] = time_to_now_batch
            return res
        else:
            instance_cnt = len(label_list)
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros((instance_cnt, max_seq_length_batch)).astype('int32')
            item_cate_history_batch_all = np.zeros((instance_cnt, max_seq_length_batch)).astype('int32')
            time_diff_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype('float32')
            time_from_first_action_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype('float32')
            time_to_now_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype('float32')
            mask = np.zeros((instance_cnt, max_seq_length_batch)).astype('float32')
            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_history_batch_all[i, :this_length] = item_history_batch[i][-this_length:]
                item_cate_history_batch_all[i, :this_length] = item_cate_history_batch[i][-this_length:]
                mask[i, :this_length] = 1.0
                time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
                time_from_first_action_batch[i, :this_length] = time_from_first_action_list[i][-this_length:]
                time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]
            res = {}
            res['labels'] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            res['users'] = np.asarray(user_list, dtype=np.float32)
            res['items'] = np.asarray(item_list, dtype=np.int32)
            res['cates'] = np.asarray(item_cate_list, dtype=np.int32)
            res['item_history'] = item_history_batch_all
            res['item_cate_history'] = item_cate_history_batch_all
            res['mask'] = mask
            res['time'] = np.asarray(time_list, dtype=np.float32)
            res['time_diff'] = time_diff_batch
            res['time_from_first_action'] = time_from_first_action_batch
            res['time_to_now'] = time_to_now_batch
            return res

    def gen_feed_dict(self, data_dict):
        if False:
            i = 10
            return i + 15
        'Construct a dictionary that maps graph elements to values.\n\n        Args:\n            data_dict (dict): A dictionary that maps string name to numpy arrays.\n\n        Returns:\n            dict: A dictionary that maps graph elements to numpy arrays.\n\n        '
        if not data_dict:
            return dict()
        feed_dict = {self.labels: data_dict['labels'], self.users: data_dict['users'], self.items: data_dict['items'], self.cates: data_dict['cates'], self.item_history: data_dict['item_history'], self.item_cate_history: data_dict['item_cate_history'], self.mask: data_dict['mask'], self.time: data_dict['time'], self.time_diff: data_dict['time_diff'], self.time_from_first_action: data_dict['time_from_first_action'], self.time_to_now: data_dict['time_to_now']}
        return feed_dict