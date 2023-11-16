"""
For machine generated datasets.
"""
import numpy as np
from neon import NervanaObject

class Task(NervanaObject):
    """
    Base class from which ticker tasks inherit.
    """

    def fetch_io(self, time_steps):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate inputs, outputs numpy tensor pair of size appropriate for this minibatch.\n\n        Arguments:\n            time_steps (int): Number of time steps in this minibatch.\n\n        Returns:\n            tuple: (input, output) tuple of numpy arrays.\n\n        '
        columns = time_steps * self.be.bsz
        inputs = np.zeros((self.nin, columns))
        outputs = np.zeros((self.nout, columns))
        return (inputs, outputs)

    def fill_buffers(self, time_steps, inputs, outputs, in_tensor, out_tensor, mask):
        if False:
            print('Hello World!')
        '\n        Prepare data for delivery to device.\n\n        Arguments:\n            time_steps (int): Number of time steps in this minibatch.\n            inputs (numpy array): Inputs numpy array\n            outputs (numpy array): Outputs numpy array\n            in_tensor (Tensor): Device buffer holding inputs\n            out_tensor (Tensor): Device buffer holding outputs\n            mask (numpy array): Device buffer for the output mask\n        '
        columns = time_steps * self.be.bsz
        inC = np.zeros((self.nin, self.max_columns))
        outC = np.zeros((self.nout, self.max_columns))
        inC[:, :columns] = inputs
        outC[:, :columns] = outputs
        in_tensor.set(inC)
        out_tensor.set(outC)
        mask[:, :columns] = 1
        mask[:, columns:] = 0

class CopyTask(Task):
    """
    Copy task from the Neural Turing Machines paper:
        http://arxiv.org/abs/1410.5401.

    This version of the task is batched.
    All sequences in the same mini-batch are the same length,
    but every new minibatch has a randomly chosen minibatch length.

    When a given minibatch has length < seq_len_max, we mask the outputs
    for time steps after time_steps_max.

    The generated data is laid out in the same way as other RNN data in neon.
    """

    def __init__(self, seq_len_max, vec_size):
        if False:
            i = 10
            return i + 15
        '\n        Set up the attributes that ticker needs to see.\n\n        Arguments:\n            seq_len_max (int): Longest allowable sequence length\n            vec_size (int): Width of the bit-vector to be copied (this was 8 in paper)\n        '
        self.seq_len_max = seq_len_max
        self.vec_size = vec_size
        self.nout = self.vec_size
        self.nin = self.vec_size + 2
        self.time_steps_func = lambda l: 2 * l + 2
        self.time_steps_max = 2 * self.seq_len_max + 2
        self.time_steps_max = self.time_steps_func(self.seq_len_max)
        self.max_columns = self.time_steps_max * self.be.bsz

    def synthesize(self, in_tensor, out_tensor, mask):
        if False:
            while True:
                i = 10
        '\n        Create a new minibatch of ticker copy task data.\n\n        Arguments:\n            in_tensor (Tensor): Device buffer holding inputs\n            out_tensor (Tensor): Device buffer holding outputs\n            mask (numpy array): Device buffer for the output mask\n        '
        seq_len = np.random.randint(1, self.seq_len_max + 1)
        time_steps = self.time_steps_func(seq_len)
        (inputs, outputs) = super(CopyTask, self).fetch_io(time_steps)
        inputs[-2, :self.be.bsz] = 1
        seq = np.random.randint(2, size=(self.vec_size, seq_len * self.be.bsz))
        stop_loc = self.be.bsz * (seq_len + 1)
        inputs[-1, stop_loc:stop_loc + self.be.bsz] = 1
        inputs[:self.vec_size, self.be.bsz:stop_loc] = seq
        outputs[:, self.be.bsz * (seq_len + 2):] = seq
        super(CopyTask, self).fill_buffers(time_steps, inputs, outputs, in_tensor, out_tensor, mask)

class RepeatCopyTask(Task):
    """
    Repeat Copy task from the Neural Turing Machines paper:
        http://arxiv.org/abs/1410.5401.

    See Also:
        See comments on :py:class:`~neon.data.ticker.CopyTask` class for more details.
    """

    def __init__(self, seq_len_max, repeat_count_max, vec_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the attributes that ticker needs to see.\n\n        Arguments:\n            seq_len_max (int): Longest allowable sequence length\n            repeat_count_max (int): Max number of repeats\n            vec_size (int): Width of the bit-vector to be copied (was 8 in paper)\n        '
        self.seq_len_max = seq_len_max
        self.repeat_count_max = seq_len_max
        self.vec_size = vec_size
        self.nout = self.vec_size + 1
        self.nin = self.vec_size + 2
        self.time_steps_func = lambda l, r: l * (r + 1) + 3
        self.time_steps_max = self.time_steps_func(self.seq_len_max, self.repeat_count_max)
        self.max_columns = self.time_steps_max * self.be.bsz

    def synthesize(self, in_tensor, out_tensor, mask):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new minibatch of ticker repeat copy task data.\n\n        Arguments:\n            in_tensor (Tensor): Device buffer holding inputs\n            out_tensor (Tensor): Device buffer holding outputs\n            mask (numpy array): Device buffer for the output mask\n        '
        seq_len = np.random.randint(1, self.seq_len_max + 1)
        repeat_count = np.random.randint(1, self.repeat_count_max + 1)
        time_steps = self.time_steps_func(seq_len, repeat_count)
        (inputs, outputs) = super(RepeatCopyTask, self).fetch_io(time_steps)
        inputs[-2, :self.be.bsz] = 1
        seq = np.random.randint(2, size=(self.vec_size, seq_len * self.be.bsz))
        stop_loc = self.be.bsz * (seq_len + 1)
        inputs[-1, stop_loc:stop_loc + self.be.bsz] = repeat_count
        inputs[:self.vec_size, self.be.bsz:stop_loc] = seq
        for i in range(repeat_count):
            start = self.be.bsz * ((i + 1) * seq_len + 2)
            stop = start + seq_len * self.be.bsz
            outputs[:-1, start:stop] = seq
        outputs[-1, -self.be.bsz:] = 1
        super(RepeatCopyTask, self).fill_buffers(time_steps, inputs, outputs, in_tensor, out_tensor, mask)

class PrioritySortTask(Task):
    """
    Priority Sort task from the Neural Turing Machines paper:
        http://arxiv.org/abs/1410.5401.

    See Also:
        See comments on :py:class:`~neon.data.ticker.CopyTask` class for more details.
    """

    def __init__(self, seq_len_max, vec_size):
        if False:
            print('Hello World!')
        '\n        Set up the attributes that ticker needs to see.\n\n        Arguments:\n            seq_len_max (int): Longest allowable sequence length\n            vec_size (int): Width of the bit-vector to be copied (this was 8 in paper)\n        '
        self.seq_len_max = seq_len_max
        self.vec_size = vec_size
        self.nout = self.vec_size
        self.nin = self.vec_size + 3
        self.time_steps_func = lambda l: 2 * l + 2
        self.time_steps_max = self.time_steps_func(self.seq_len_max)
        self.max_columns = self.time_steps_max * self.be.bsz

    def synthesize(self, in_tensor, out_tensor, mask):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new minibatch of ticker priority sort task data.\n\n        Arguments:\n            in_tensor: device buffer holding inputs\n            out_tensor: device buffer holding outputs\n            mask: device buffer for the output mask\n        '
        seq_len = np.random.randint(1, self.seq_len_max + 1)
        time_steps = self.time_steps_func(seq_len)
        (inputs, outputs) = super(PrioritySortTask, self).fetch_io(time_steps)
        inputs[-3, :self.be.bsz] = 1
        seq = np.random.randint(2, size=(self.nin, seq_len * self.be.bsz)).astype(float)
        seq[-3:, :] = 0
        priorities = np.random.uniform(-1, 1, size=(seq_len * self.be.bsz,))
        seq[-1, :] = priorities
        stop_loc = self.be.bsz * (seq_len + 1)
        inputs[-2, stop_loc:stop_loc + self.be.bsz] = 1
        inputs[:, self.be.bsz:stop_loc] = seq
        for i in range(self.be.bsz):
            x = seq[:, i::self.be.bsz]
            x = x[:, x[-1, :].argsort()]
            seq[:, i::self.be.bsz] = x
        outputs[:, self.be.bsz * (seq_len + 2):] = seq[:self.nout, :]
        super(PrioritySortTask, self).fill_buffers(time_steps, inputs, outputs, in_tensor, out_tensor, mask)

class Ticker(NervanaObject):
    """
    This class defines methods for generating and iterating over ticker datasets.
    """

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reset has no meaning in the context of ticker data.\n        '
        pass

    def __init__(self, task):
        if False:
            i = 10
            return i + 15
        '\n        Construct a ticker dataset object.\n\n        Arguments:\n            task: An object representing the task to be trained on\n                  It contains information about input and output size,\n                  sequence length, etc. It also implements a synthesize function,\n                  which is used to generate the next minibatch of data.\n        '
        self.task = task
        self.batch_index = 0
        self.nbatches = 100
        self.ndata = self.nbatches * self.be.bsz
        self.nout = task.nout
        self.nin = task.nin
        self.shape = (self.nin, self.task.time_steps_max)
        self.dev_X = self.be.iobuf((self.nin, self.task.time_steps_max))
        self.dev_y = self.be.iobuf((self.nout, self.task.time_steps_max))
        self.mask = self.be.iobuf((self.nout, self.task.time_steps_max))

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        '\n        Generator that can be used to iterate over this dataset.\n\n        Yields:\n            tuple : the next minibatch of data.\n\n        Note:\n            The second element of the tuple is itself a tuple (t,m) with:\n                t: the actual target as generated by the task object\n                m: the output mask to account for the difference between\n                    the seq_length for this minibatch and the max seq_len,\n                    which is also the number of columns in X,t, and m\n        '
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            self.task.synthesize(self.dev_X, self.dev_y, self.mask)
            self.batch_index += 1
            yield (self.dev_X, (self.dev_y, self.mask))