import numpy as np
from ludwig.data.batcher.base import Batcher

class BucketedBatcher(Batcher):

    def __init__(self, dataset, bucketing_field, batch_size=128, buckets=10, should_shuffle=True, ignore_last=False, should_trim=False, trim_side='right'):
        if False:
            while True:
                i = 10
        self.should_shuffle = should_shuffle
        self.bucketing_field = bucketing_field
        self.should_trim = should_trim
        self.trim_side = trim_side
        self.dataset = dataset
        field = dataset.get_dataset()[bucketing_field]
        field_lengths = np.apply_along_axis(lambda x: np.sign(x).sum(), 1, field)
        sorted_idcs = np.argsort(field_lengths)
        self.buckets_idcs = []
        datapoints_per_bucket = len(field) // buckets
        for b in range(buckets):
            start = datapoints_per_bucket * b
            end = datapoints_per_bucket * (b + 1) if b < buckets - 1 else len(sorted_idcs)
            self.buckets_idcs.append(sorted_idcs[start:end])
        if should_shuffle:
            self.shuffle(self.buckets_idcs)
        self.ignore_last = ignore_last
        self.batch_size = batch_size
        self.total_size = min(map(len, dataset.get_dataset().values()))
        self.bucket_sizes = np.array([x for x in map(len, self.buckets_idcs)])
        self.steps_per_epoch = self._compute_steps_per_epoch()
        self.indices = np.array([0] * buckets)
        self.step = 0
        self.epoch = 0

    def shuffle(self, buckets_idcs):
        if False:
            print('Hello World!')
        for i in range(len(buckets_idcs)):
            np.random.shuffle(buckets_idcs[i])

    def next_batch(self):
        if False:
            while True:
                i = 10
        if self.last_batch():
            if self.should_shuffle:
                self.shuffle(self.buckets_idcs)
            self.set_epoch(self.epoch + 1)
        if self.ignore_last:
            idcs_below_size = self.indices + self.batch_size < self.bucket_sizes
        else:
            idcs_below_size = self.indices < self.bucket_sizes
        i = np.random.choice(np.arange(0, len(self.buckets_idcs))[idcs_below_size])
        selected_bucket = self.buckets_idcs[i]
        selected_idcs = selected_bucket[self.indices[i]:self.indices[i] + self.batch_size]
        sub_batch = {}
        for key in self.dataset.get_dataset():
            if key == self.bucketing_field and self.should_trim:
                selected_samples = self.dataset.get(key, selected_idcs)
                max_length = np.sign(selected_samples).sum(axis=1).max()
                if self.trim_side == 'right':
                    sub_batch[key] = selected_samples[:, :max_length]
                elif self.trim_side == 'left':
                    sub_batch[key] = selected_samples[:, -max_length:]
                else:
                    raise ValueError('Invalid trim side:', self.trim_side)
            else:
                sub_batch[key] = self.dataset.get(key, selected_idcs)
        self.indices[i] += self.batch_size
        self.step += 1
        return sub_batch

    def last_batch(self):
        if False:
            for i in range(10):
                print('nop')
        return not np.any(self.indices < self.bucket_sizes) or (self.ignore_last and (not np.any(self.indices + self.batch_size < self.bucket_sizes)))

    def set_epoch(self, epoch, batch_size):
        if False:
            for i in range(10):
                print('nop')
        self.indices = np.array([0] * len(self.buckets_idcs))
        self.step = 0
        self.epoch = epoch
        self.batch_size = batch_size
        self.steps_per_epoch = self._compute_steps_per_epoch()

    def _compute_steps_per_epoch(self) -> int:
        if False:
            while True:
                i = 10
        return int(np.sum(np.ceil(self.bucket_sizes / self.batch_size)).item())