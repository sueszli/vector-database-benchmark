import os
import abc
import numpy as np
import tensorflow as tf
from recommenders.models.deeprec.models.base_model import BaseModel
from recommenders.models.deeprec.deeprec_utils import cal_metric, load_dict
__all__ = ['SequentialBaseModel']

class SequentialBaseModel(BaseModel):
    """Base class for sequential models"""

    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        if False:
            while True:
                i = 10
        'Initializing the model. Create common logics which are needed by all sequential models, such as loss function,\n        parameter set.\n\n        Args:\n            hparams (HParams): A `HParams` object, hold the entire set of hyperparameters.\n            iterator_creator (object): An iterator to load the data.\n            graph (object): An optional graph.\n            seed (int): Random seed.\n        '
        self.hparams = hparams
        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError('Please confirm the number of negative samples for each positive instance.')
        self.min_seq_length = hparams.min_seq_length if 'min_seq_length' in hparams.values() else 1
        self.hidden_size = hparams.hidden_size if 'hidden_size' in hparams.values() else None
        self.graph = tf.Graph() if not graph else graph
        with self.graph.as_default():
            self.sequence_length = tf.compat.v1.placeholder(tf.int32, [None], name='sequence_length')
        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        if False:
            while True:
                i = 10
        'Subclass will implement this.'
        pass

    def _build_graph(self):
        if False:
            print('Hello World!')
        'The main function to create sequential models.\n\n        Returns:\n            object: the prediction score make by the model.\n        '
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        with tf.compat.v1.variable_scope('sequential') as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_output = self._build_seq_graph()
            logit = self._fcn_net(model_output, hparams.layer_sizes, scope='logit_fcn')
            self._add_norm()
            return logit

    def fit(self, train_file, valid_file, valid_num_ngs, eval_metric='group_auc'):
        if False:
            while True:
                i = 10
        'Fit the model with `train_file`. Evaluate the model on `valid_file` per epoch to observe the training status.\n        If `test_file` is not None, evaluate it too.\n\n        Args:\n            train_file (str): training data set.\n            valid_file (str): validation set.\n            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.\n            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.\n\n        Returns:\n            object: An instance of self.\n        '
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError('Please specify a positive integer of negative numbers for training without sampling needed.')
        if valid_num_ngs < 1:
            raise ValueError('Please specify a positive integer of negative numbers for validation.')
        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1
        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)
            self.writer = tf.compat.v1.summary.FileWriter(self.hparams.SUMMARIES_DIR, self.sess.graph)
        train_sess = self.sess
        eval_info = list()
        (best_metric, self.best_epoch) = (0, 0)
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            file_iterator = self.iterator.load_data_from_file(train_file, min_seq_length=self.min_seq_length, batch_num_ngs=self.train_num_ngs)
            for batch_data_input in file_iterator:
                if batch_data_input:
                    step_result = self.train(train_sess, batch_data_input)
                    (_, _, step_loss, step_data_loss, summary) = step_result
                    if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                        self.writer.add_summary(summary, step)
                    epoch_loss += step_loss
                    step += 1
                    if step % self.hparams.show_step == 0:
                        print('step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}'.format(step, step_loss, step_data_loss))
            valid_res = self.run_eval(valid_file, valid_num_ngs)
            print('eval valid at epoch {0}: {1}'.format(epoch, ','.join(['' + str(key) + ':' + str(value) for (key, value) in valid_res.items()])))
            eval_info.append((epoch, valid_res))
            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            elif early_stop > 0 and epoch - self.best_epoch >= early_stop:
                print('early stop at epoch {0}!'.format(epoch))
                break
            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(sess=train_sess, save_path=self.hparams.MODEL_DIR + 'epoch_' + str(epoch))
                    checkpoint_path = self.saver.save(sess=train_sess, save_path=os.path.join(self.hparams.MODEL_DIR, 'best_model'))
        if self.hparams.write_tfevents:
            self.writer.close()
        print(eval_info)
        print('best epoch: {0}'.format(self.best_epoch))
        return self

    def run_eval(self, filename, num_ngs):
        if False:
            print('Hello World!')
        'Evaluate the given file and returns some evaluation metrics.\n\n        Args:\n            filename (str): A file name that will be evaluated.\n            num_ngs (int): The number of negative sampling for a positive instance.\n\n        Returns:\n            dict: A dictionary that contains evaluation metrics.\n        '
        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        for batch_data_input in self.iterator.load_data_from_file(filename, min_seq_length=self.min_seq_length, batch_num_ngs=0):
            if batch_data_input:
                (step_pred, step_labels) = self.eval(load_sess, batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))
        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(group_labels, group_preds, self.hparams.pairwise_metrics)
        res.update(res_pairwise)
        return res

    def predict(self, infile_name, outfile_name):
        if False:
            while True:
                i = 10
        'Make predictions on the given data, and output predicted scores to a file.\n\n        Args:\n            infile_name (str): Input file name.\n            outfile_name (str): Output file name.\n\n        Returns:\n            object: An instance of self.\n        '
        load_sess = self.sess
        with tf.io.gfile.GFile(outfile_name, 'w') as wt:
            for batch_data_input in self.iterator.load_data_from_file(infile_name, batch_num_ngs=0):
                if batch_data_input:
                    step_pred = self.infer(load_sess, batch_data_input)
                    step_pred = np.reshape(step_pred, -1)
                    wt.write('\n'.join(map(str, step_pred)))
                    wt.write('\n')
        return self

    def _build_embedding(self):
        if False:
            return 10
        'The field embedding layer. Initialization of embedding variables.'
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim
        with tf.compat.v1.variable_scope('embedding', initializer=self.initializer):
            self.user_lookup = tf.compat.v1.get_variable(name='user_embedding', shape=[self.user_vocab_length, self.user_embedding_dim], dtype=tf.float32)
            self.item_lookup = tf.compat.v1.get_variable(name='item_embedding', shape=[self.item_vocab_length, self.item_embedding_dim], dtype=tf.float32)
            self.cate_lookup = tf.compat.v1.get_variable(name='cate_embedding', shape=[self.cate_vocab_length, self.cate_embedding_dim], dtype=tf.float32)

    def _lookup_from_embedding(self):
        if False:
            while True:
                i = 10
        'Lookup from embedding variables. A dropout layer follows lookup operations.'
        self.user_embedding = tf.nn.embedding_lookup(params=self.user_lookup, ids=self.iterator.users)
        tf.compat.v1.summary.histogram('user_embedding_output', self.user_embedding)
        self.item_embedding = tf.compat.v1.nn.embedding_lookup(params=self.item_lookup, ids=self.iterator.items)
        self.item_history_embedding = tf.compat.v1.nn.embedding_lookup(params=self.item_lookup, ids=self.iterator.item_history)
        tf.compat.v1.summary.histogram('item_history_embedding_output', self.item_history_embedding)
        self.cate_embedding = tf.compat.v1.nn.embedding_lookup(params=self.cate_lookup, ids=self.iterator.cates)
        self.cate_history_embedding = tf.compat.v1.nn.embedding_lookup(params=self.cate_lookup, ids=self.iterator.item_cate_history)
        tf.compat.v1.summary.histogram('cate_history_embedding_output', self.cate_history_embedding)
        involved_items = tf.concat([tf.reshape(self.iterator.item_history, [-1]), tf.reshape(self.iterator.items, [-1])], -1)
        (self.involved_items, _) = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(params=self.item_lookup, ids=self.involved_items)
        self.embed_params.append(involved_item_embedding)
        involved_cates = tf.concat([tf.reshape(self.iterator.item_cate_history, [-1]), tf.reshape(self.iterator.cates, [-1])], -1)
        (self.involved_cates, _) = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(params=self.cate_lookup, ids=self.involved_cates)
        self.embed_params.append(involved_cate_embedding)
        self.target_item_embedding = tf.concat([self.item_embedding, self.cate_embedding], -1)
        tf.compat.v1.summary.histogram('target_item_embedding_output', self.target_item_embedding)

    def _add_norm(self):
        if False:
            return 10
        'Regularization for embedding variables and other variables.'
        (all_variables, embed_variables) = (tf.compat.v1.trainable_variables(), tf.compat.v1.trainable_variables(self.sequential_scope._name + '/embedding'))
        layer_params = list(set(all_variables) - set(embed_variables))
        layer_params = [a for a in layer_params if '_no_reg' not in a.name]
        self.layer_params.extend(layer_params)