from __future__ import division
import numpy as np
import h5py
import copy
from neon.data.dataiterator import NervanaDataIterator

class SentenceEncode(NervanaDataIterator):
    """
    This class defines an iterator for loading and iterating the sentences in
    a structure to encode sentences into the skip-thought vectors
    """

    def __init__(self, sentences, sentence_text, nsamples, nwords, max_len=100, index_from=2):
        if False:
            while True:
                i = 10
        '\n        Construct a sentence dataset object.\n        Build the context using skip-thought model\n\n        Aguments:\n            sentences: list of tokenized (and int-encoded) sentences to use for iteration\n            sentence_text: list of raw text sentences\n            nsamples: number of sentences\n            nwords: number of words in vocab\n        '
        super(SentenceEncode, self).__init__(name=None)
        self.nsamples = nsamples
        self.nwords = nwords
        self.batch_index = 0
        self.nbatches = 0
        self.max_len = max_len
        self.index_from = index_from
        source = sentences[:nsamples]
        source_text = sentence_text[:nsamples]
        extra_sent = len(source) % self.be.bsz
        self.nbatches = len(source) // self.be.bsz
        self.ndata = self.nbatches * self.be.bsz
        if extra_sent:
            source = source[:-extra_sent]
            source_text = source_text[:-extra_sent]
        self.sent_len = dict(((i, min(len(c), self.max_len)) for (i, c) in enumerate(source)))
        self.X = source
        self.X_text = source_text
        self.dev_X = self.be.iobuf(self.max_len, dtype=np.int32)
        self.X_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.shape = (self.max_len, 1)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        For resetting the starting index of this dataset back to zero.\n        Relevant for when one wants to call repeated evaluations on the dataset\n        but don't want to wrap around for the last uneven minibatch\n        Not necessary when ndata is divisible by batch size\n        "
        self.batch_index = 0

    def __iter__(self):
        if False:
            while True:
                i = 10
        '\n        Generator that can be used to iterate over this dataset\n        '
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            self.X_np.fill(0)
            idx = range(self.batch_index * self.be.bsz, (self.batch_index + 1) * self.be.bsz)
            for (i, ix) in enumerate(idx):
                s_len = self.sent_len[ix]
                self.X_np[-s_len:, i] = self.X[ix][-s_len:] + self.index_from
            self.dev_X.set(self.X_np)
            self.batch_index += 1
            yield (self.dev_X, None)

class SentenceHomogenous(NervanaDataIterator):
    """
    This class defines an iterator for loading and iterating the sentences in
    a structure to train the skip-thought vectors
    """

    def __init__(self, data_file=None, sent_name=None, text_name=None, nwords=None, max_len=30, index_from=2, eos=3):
        if False:
            i = 10
            return i + 15
        '\n        Construct a sentence dataset object.\n        Build the context using skip-thought model\n\n        Aguments:\n            data_file (str): path to hdf5 file containing sentences\n            sent_name (str): name of tokenized dataset\n            text_name (str): name of raw text dataset\n            nwords (int): size of vocabulary\n            max_len (int): maximum number of words per sentence\n            index_from (int): index offset for padding (0) and OOV (1)\n            eos (int): index of EOS token\n        '
        super(SentenceHomogenous, self).__init__(name=None)
        self.nwords = nwords
        self.batch_index = 0
        self.nbatches = 0
        self.max_len = max_len
        self.index_from = index_from
        self.eos = eos
        self.data_file = data_file
        self.sent_name = sent_name
        self.text_name = text_name
        h5f = h5py.File(self.data_file, 'r+')
        sentences = h5f[self.sent_name][:]
        self.nsamples = h5f[self.sent_name].attrs['nsample'] - 2
        self.source = sentences[1:-1]
        self.forward = sentences[2:]
        self.backward = sentences[:-2]
        self.lengths = [len(cc) for cc in self.source]
        self.len_unique = np.unique(self.lengths)
        self.len_unique = [ll for ll in self.len_unique if ll <= self.max_len]
        self.len_indicies = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indicies[ll] = np.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indicies[ll])
        self.nbatches = 0
        for ll in self.len_unique:
            self.nbatches += int(np.ceil(self.len_counts[ll] / float(self.be.bsz)))
        self.ndata = self.nbatches * self.be.bsz
        self.len_curr_counts = copy.copy(self.len_counts)
        self.dev_X = self.be.iobuf(self.max_len, dtype=np.int32)
        self.dev_X_p = self.be.iobuf(self.max_len, dtype=np.int32)
        self.dev_X_n = self.be.iobuf(self.max_len, dtype=np.int32)
        self.X_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.X_p_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.X_n_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.dev_y_p_flat = self.be.iobuf((1, self.max_len), dtype=np.int32)
        self.dev_y_n_flat = self.be.iobuf((1, self.max_len), dtype=np.int32)
        self.dev_y_p = self.be.iobuf((nwords, self.max_len), dtype=np.int32)
        self.dev_y_n = self.be.iobuf((nwords, self.max_len), dtype=np.int32)
        self.dev_y_p_mask = self.be.iobuf((nwords, self.max_len), dtype=np.int32)
        self.dev_y_n_mask = self.be.iobuf((nwords, self.max_len), dtype=np.int32)
        self.dev_y_p_mask_list = self.get_bsz(self.dev_y_p_mask, self.max_len)
        self.dev_y_n_mask_list = self.get_bsz(self.dev_y_n_mask, self.max_len)
        self.y_p_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.y_n_np = np.empty((self.max_len, self.be.bsz), dtype=np.int32)
        self.clear_list = [self.X_np, self.X_p_np, self.X_n_np, self.y_p_np, self.y_n_np, self.dev_y_p_mask, self.dev_y_n_mask]
        self.shape = [(self.max_len, 1), (self.max_len, 1), (self.max_len, 1)]
        h5f.close()
        self.reset()

    def reset(self):
        if False:
            while True:
                i = 10
        "\n        For resetting the starting index of this dataset back to zero.\n        Relevant for when one wants to call repeated evaluations on the dataset\n        but don't want to wrap around for the last uneven minibatch\n        Not necessary when ndata is divisible by batch size\n        "
        self.batch_index = 0
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = np.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indicies[ll] = np.random.permutation(self.len_indicies[ll])
        self.len_idx = -1

    def next(self):
        if False:
            i = 10
            return i + 15
        '\n        Method called by iterator to get a new batch of sentence triplets:\n        (source, forward, backward). Sentences are returned in order of increasing length,\n        and source sentences of each batch all have the same length.\n        '
        self.clear_device_buffer()
        count = 0
        while True:
            self.len_idx = np.mod(self.len_idx + 1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()
        curr_len = self.len_unique[self.len_idx]
        curr_batch_size = np.minimum(self.be.bsz, self.len_curr_counts[curr_len])
        curr_pos = self.len_indices_pos[curr_len]
        curr_indices = self.len_indicies[curr_len][curr_pos:curr_pos + curr_batch_size]
        self.len_indices_pos[curr_len] += curr_batch_size
        self.len_curr_counts[curr_len] -= curr_batch_size
        source_batch = [self.source[ii] for ii in curr_indices]
        forward_batch = [self.forward[ii] for ii in curr_indices]
        backward_batch = [self.backward[ii] for ii in curr_indices]
        for i in range(len(source_batch)):
            l_s = min(len(source_batch[i]), self.max_len)
            if len(source_batch[i][-l_s:]) == 0:
                continue
            self.X_np[-l_s:, i] = source_batch[i][-l_s:] + self.index_from
            l_p = min(len(backward_batch[i]), self.max_len)
            self.X_p_np[:l_p, i] = [self.eos] + (backward_batch[i][-l_p:-1] + self.index_from).tolist()
            self.y_p_np[:l_p, i] = backward_batch[i][-l_p:] + self.index_from
            self.dev_y_p_mask_list[i][:, :l_p] = 1
            l_n = min(len(forward_batch[i]), self.max_len)
            self.X_n_np[:l_n, i] = [self.eos] + (forward_batch[i][-l_n:-1] + self.index_from).tolist()
            self.y_n_np[:l_n, i] = forward_batch[i][-l_n:] + self.index_from
            self.dev_y_n_mask_list[i][:, :l_n] = 1
        self.dev_X.set(self.X_np)
        self.dev_X_p.set(self.X_p_np)
        self.dev_X_n.set(self.X_n_np)
        self.dev_y_p_flat.set(self.y_p_np.reshape(1, -1))
        self.dev_y_n_flat.set(self.y_n_np.reshape(1, -1))
        self.dev_y_p[:] = self.be.onehot(self.dev_y_p_flat, axis=0)
        self.dev_y_n[:] = self.be.onehot(self.dev_y_n_flat, axis=0)
        self.batch_index += 1
        return ((self.dev_X, self.dev_X_p, self.dev_X_n), ((self.dev_y_p, self.dev_y_p_mask), (self.dev_y_n, self.dev_y_n_mask)))

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generator that can be used to iterate over this dataset\n        Input: clip a long sentence from the left\n               encoder input: take sentence and pad 0 from the left\n               decoder input: take the sentence length -1, prepend a <eos>, pad 0 from the right\n        output: decoder output: take the sentence length, pad 0 from the right\n        '
        return self

    def clear_device_buffer(self):
        if False:
            return 10
        ' Clear the buffers used to hold batches. '
        if self.clear_list:
            [dev.fill(0) for dev in self.clear_list]

    def get_bsz(self, x, nsteps):
        if False:
            return 10
        if x is None:
            return [None for b in range(self.be.bsz)]
        xs = x.reshape(-1, nsteps, self.be.bsz)
        return [xs[:, :, b] for b in range(self.be.bsz)]
    __next__ = next