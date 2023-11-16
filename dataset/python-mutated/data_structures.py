"""Data structures used by the speech client."""

class RollingMean:
    """Simple rolling mean calculation optimized for speed.

    The optimization is made for cases where value retrieval is made at a
    comparative rate to the sample additions.

    Args:
        mean_samples: Number of samples to use for mean value
    """

    def __init__(self, mean_samples):
        if False:
            print('Hello World!')
        self.num_samples = mean_samples
        self.samples = []
        self.value = None
        self.replace_pos = 0

    def append_sample(self, sample):
        if False:
            while True:
                i = 10
        'Add a sample to the buffer.\n\n        The sample will be appended if there is room in the buffer,\n        otherwise it will replace the oldest sample in the buffer.\n        '
        sample = float(sample)
        current_len = len(self.samples)
        if current_len < self.num_samples:
            self.samples.append(sample)
            if self.value is not None:
                avgsum = self.value * current_len + sample
                self.value = avgsum / (current_len + 1)
            else:
                self.value = sample
        else:
            replace_val = self.samples[self.replace_pos]
            self.value -= replace_val / self.num_samples
            self.value += sample / self.num_samples
            self.samples[self.replace_pos] = sample
            self.replace_pos = (self.replace_pos + 1) % self.num_samples

class CyclicAudioBuffer:
    """A Cyclic audio buffer for storing binary data.

    TODO: The class is still unoptimized and performance can probably be
    enhanced.

    Args:
        size (int): size in bytes
        initial_data (bytes): initial buffer data
    """

    def __init__(self, size, initial_data):
        if False:
            print('Hello World!')
        self.size = size
        self._buffer = initial_data[-size:]

    def append(self, data):
        if False:
            while True:
                i = 10
        'Add new data to the buffer, and slide out data if the buffer is full\n\n        Args:\n            data (bytes): binary data to append to the buffer. If buffer size\n                          is exceeded the oldest data will be dropped.\n        '
        buff = self._buffer + data
        if len(buff) > self.size:
            buff = buff[-self.size:]
        self._buffer = buff

    def get(self):
        if False:
            i = 10
            return i + 15
        'Get the binary data.'
        return self._buffer

    def get_last(self, size):
        if False:
            return 10
        'Get the last entries of the buffer.'
        return self._buffer[-size:]

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self._buffer[key]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._buffer)