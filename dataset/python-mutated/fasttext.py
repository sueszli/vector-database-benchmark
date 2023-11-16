"""
Text classification with fastText models that are compatible with cleanlab.
This module allows you to easily find label issues in your text datasets.

You must have fastText installed: ``pip install fasttext``.

Tips:

* Check out our example using this class: `fasttext_amazon_reviews <https://github.com/cleanlab/examples/blob/master/fasttext_amazon_reviews/fasttext_amazon_reviews.ipynb>`_
* Our `unit tests <https://github.com/cleanlab/cleanlab/blob/master/tests/test_frameworks.py>`_ also provide basic usage examples.

"""
import time
import os
import copy
import numpy as np
from sklearn.base import BaseEstimator
from fasttext import train_supervised, load_model
LABEL = '__label__'
NEWLINE = ' __newline__ '

def data_loader(fn=None, indices=None, label=LABEL, batch_size=1000):
    if False:
        while True:
            i = 10
    'Returns a generator, yielding two lists containing\n    [labels], [text]. Items are always returned in the\n    order in the file, regardless if indices are provided.'

    def _split_labels_and_text(batch):
        if False:
            for i in range(10):
                print('nop')
        (l, t) = [list(t) for t in zip(*(z.split(' ', 1) for z in batch))]
        return (l, t)
    if indices is not None:
        stack_indices = sorted(indices, reverse=True)
        stack_idx = stack_indices.pop()
    with open(fn, 'r') as f:
        len_label = len(label)
        idx = 0
        batch_counter = 0
        prev = f.readline()
        batch = []
        while True:
            try:
                line = f.readline()
                line = line
                if line[:len_label] == label or line == '':
                    if indices is None or stack_idx == idx:
                        batch.append(prev.strip().replace('\n', NEWLINE))
                        batch_counter += 1
                        if indices is not None:
                            if len(stack_indices):
                                stack_idx = stack_indices.pop()
                            else:
                                yield _split_labels_and_text(batch)
                                break
                    prev = ''
                    idx += 1
                    if batch_counter == batch_size:
                        yield _split_labels_and_text(batch)
                        batch_counter = 0
                        batch = []
                prev += line
                if line == '':
                    if len(batch) > 0:
                        yield _split_labels_and_text(batch)
                    break
            except EOFError:
                if indices is None or stack_idx == idx:
                    batch.append(prev.strip().replace('\n', NEWLINE))
                    batch_counter += 1
                    yield _split_labels_and_text(batch)
                break

class FastTextClassifier(BaseEstimator):
    """Instantiate a fastText classifier that is compatible with :py:class:`CleanLearning <cleanlab.classification.CleanLearning>`.

    Parameters
    ----------
    train_data_fn: str
        File name of the training data in the format compatible with fastText.

    test_data_fn: str, optional
        File name of the test data in the format compatible with fastText.
    """

    def __init__(self, train_data_fn, test_data_fn=None, labels=None, tmp_dir='', label=LABEL, del_intermediate_data=True, kwargs_train_supervised={}, p_at_k=1, batch_size=1000):
        if False:
            print('Hello World!')
        self.train_data_fn = train_data_fn
        self.test_data_fn = test_data_fn
        self.tmp_dir = tmp_dir
        self.label = label
        self.del_intermediate_data = del_intermediate_data
        self.kwargs_train_supervised = kwargs_train_supervised
        self.p_at_k = p_at_k
        self.batch_size = batch_size
        self.clf = None
        self.labels = labels
        if labels is None:
            unique_labels = set([])
            for (labels, _) in data_loader(fn=train_data_fn, batch_size=batch_size):
                unique_labels = unique_labels.union(set(labels))
            if test_data_fn is not None:
                for (labels, _) in data_loader(fn=test_data_fn, batch_size=batch_size):
                    unique_labels = unique_labels.union(set(labels))
        else:
            unique_labels = [label + str(l) for l in labels]
        unique_labels = sorted(list(unique_labels))
        self.label2num = dict(zip(unique_labels, range(len(unique_labels))))
        self.num2label = dict(((y, x) for (x, y) in self.label2num.items()))

    def _create_train_data(self, data_indices):
        if False:
            while True:
                i = 10
        'Returns filename of the masked fasttext data file.\n        Items are written in the order they are in the file,\n        regardless if indices are provided.'
        if data_indices is None:
            self.masked_data_was_created = False
            return self.train_data_fn
        else:
            len_label = len(LABEL)
            data_indices = sorted(data_indices, reverse=True)
            masked_fn = 'fastTextClf_' + str(int(time.time())) + '.txt'
            open(masked_fn, 'w').close()
            with open(self.train_data_fn, 'r') as rf:
                idx = 0
                data_idx = data_indices.pop()
                for line in rf:
                    if idx == data_idx:
                        with open(masked_fn, 'a') as wf:
                            wf.write(line.strip().replace('\n', NEWLINE) + '\n')
                        if line[:len_label] == LABEL:
                            if len(data_indices):
                                data_idx = data_indices.pop()
                            else:
                                break
                    if line[:len_label] == LABEL:
                        idx += 1
            self.masked_data_was_created = True
        return masked_fn

    def _remove_masked_data(self, fn):
        if False:
            print('Hello World!')
        'Deletes intermediate data files.'
        if self.del_intermediate_data and self.masked_data_was_created:
            os.remove(fn)

    def __deepcopy__(self, memo):
        if False:
            while True:
                i = 10
        if self.clf is None:
            self_clf_copy = None
        else:
            fn = 'tmp_{}.fasttext.model'.format(int(time.time()))
            self.clf.save_model(fn)
            self_clf_copy = load_model(fn)
            os.remove(fn)
        params = self.__dict__
        clf = params.pop('clf')
        params_copy = copy.deepcopy(params)
        self.clf = clf
        clf_copy = FastTextClassifier(self.train_data_fn)
        params_copy['clf'] = self_clf_copy
        clf_copy.__dict__ = params_copy
        return clf_copy

    def fit(self, X=None, y=None, sample_weight=None):
        if False:
            print('Hello World!')
        "Trains the fast text classifier.\n        Typical usage requires NO parameters,\n        just clf.fit()  # No params.\n\n        Parameters\n        ----------\n        X : iterable, e.g. list, numpy array (default None)\n          The list of indices of the data to use.\n          When in doubt, set as None. None defaults to range(len(data)).\n        y : None\n          Leave this as None. It's a filler to suit sklearns reqs.\n        sample_weight : None\n          Leave this as None. It's a filler to suit sklearns reqs."
        train_fn = self._create_train_data(data_indices=X)
        self.clf = train_supervised(train_fn, **self.kwargs_train_supervised)
        self._remove_masked_data(train_fn)

    def predict_proba(self, X=None, train_data=True, return_labels=False):
        if False:
            i = 10
            return i + 15
        'Produces a probability matrix with examples on rows and\n        classes on columns, where each row sums to 1 and captures the\n        probability of the example belonging to each class.'
        fn = self.train_data_fn if train_data else self.test_data_fn
        pred_probs_list = []
        if return_labels:
            labels_list = []
        for (labels, text) in data_loader(fn=fn, indices=X, batch_size=self.batch_size):
            pred = self.clf.predict(text=text, k=len(self.clf.get_labels()))
            pred_probs = [[p for (_, p) in sorted(list(zip(*l)), key=lambda x: x[0])] for l in list(zip(*pred))]
            pred_probs_list.append(np.array(pred_probs))
            if return_labels:
                labels_list.append(labels)
        pred_probs = np.concatenate(pred_probs_list, axis=0)
        if return_labels:
            gold_labels = [self.label2num[z] for l in labels_list for z in l]
            return (pred_probs, np.array(gold_labels))
        else:
            return pred_probs

    def predict(self, X=None, train_data=True, return_labels=False):
        if False:
            for i in range(10):
                print('nop')
        'Predict labels of X'
        fn = self.train_data_fn if train_data else self.test_data_fn
        pred_list = []
        if return_labels:
            labels_list = []
        for (labels, text) in data_loader(fn=fn, indices=X, batch_size=self.batch_size):
            pred = [self.label2num[z[0]] for z in self.clf.predict(text)[0]]
            pred_list.append(pred)
            if return_labels:
                labels_list.append(labels)
        pred = np.array([z for l in pred_list for z in l])
        if return_labels:
            gold_labels = [self.label2num[z] for l in labels_list for z in l]
            return (pred, np.array(gold_labels))
        else:
            return pred

    def score(self, X=None, y=None, sample_weight=None, k=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute the average precision @ k (single label) of the\n        labels predicted from X and the true labels given by y.\n        score expects a `y` variable. In this case, `y` is the noisy labels.'
        if k is None:
            k = self.p_at_k
        fn = self.test_data_fn
        pred_list = []
        if y is None:
            labels_list = []
        for (labels, text) in data_loader(fn=fn, indices=X, batch_size=self.batch_size):
            pred = self.clf.predict(text, k=k)[0]
            pred_list.append(pred)
            if y is None:
                labels_list.append(labels)
        pred = np.array([z for l in pred_list for z in l])
        if y is None:
            y = [z for l in labels_list for z in l]
        else:
            y = [self.num2label[z] for z in y]
        apk = np.mean([y[i] in l for (i, l) in enumerate(pred)])
        return apk