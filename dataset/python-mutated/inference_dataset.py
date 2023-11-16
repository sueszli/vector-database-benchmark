import torch

class InferenceDataset:

    def __init__(self, dataset, prefix, only_prefix=True, presort_by_length=True, filter_short=False, min_length=None):
        if False:
            while True:
                i = 10
        self.dataset = dataset
        self.collater = self.dataset.collater
        self.prefix = prefix
        self.only_prefix = only_prefix
        self.filter_short = filter_short
        self.remapping = list(range(len(self.dataset)))
        if min_length:
            assert min_length >= prefix + 1
        length_thr = prefix + 1 if not min_length else min_length
        if filter_short:
            self.remapping = list(filter(lambda i: self.dataset[i]['dur_source'].sum() > length_thr, self.remapping))
            print(f'# the initial dataset of {len(self.dataset)} examples became {len(self.remapping)} after filtering examples shorter than {length_thr} (in duration units)')
        if presort_by_length:
            lengths = {index: dataset.size(index) for index in self.remapping}
            self.remapping.sort(key=lambda i: lengths[i])

    @property
    def pads(self):
        if False:
            print('Hello World!')
        return self.dataset.pads

    def __len__(self):
        if False:
            return 10
        return len(self.remapping)

    def original_size(self, k):
        if False:
            i = 10
            return i + 15
        k = self.remapping[k]
        return self.dataset.size(k)

    def __getitem__(self, k):
        if False:
            return 10
        k = self.remapping[k]
        channels = self.dataset[k]
        if self.prefix and self.only_prefix:
            dur_channel = channels['dur_source']
            assert dur_channel.sum() >= self.prefix
            token_times = dur_channel.cumsum(dim=-1)
            cut_after = torch.searchsorted(token_times, torch.tensor(self.prefix))
            r = {}
            for (channel_name, value) in channels.items():
                if isinstance(value, torch.Tensor) and 'source' in channel_name:
                    r[channel_name] = value[:cut_after + 1]
                else:
                    r[channel_name] = value
            r['prefix'] = cut_after + 1
        else:
            r = channels
        return r

def explode_batch(batch, times):
    if False:
        return 10
    if times == 1:
        return batch
    new_batch = {}
    for (key, value) in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.size(0) == 1
            new_batch[key] = torch.cat([value] * times)
        elif key in ['ntokens', 'nsentences']:
            new_batch[key] = value * times
        elif key in ['prefix', 'filename']:
            new_batch[key] = value
        elif key == 'net_input':
            new_batch[key] = explode_batch(value, times)
        else:
            assert False, key
    return new_batch