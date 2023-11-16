import json

class LazyDataset(object):
    """
    Lazy load dataset from disk.

    Each line of data file is a preprocessed example.
    """

    def __init__(self, data_file, reader, transform=lambda s: json.loads(s)):
        if False:
            print('Hello World!')
        '\n        Initialize lazy dataset.\n\n        By default, loading .jsonl format.\n\n        :param data_file\n        :type str\n\n        :param transform\n        :type callable\n        '
        self.data_file = data_file
        self.transform = transform
        self.reader = reader
        self.offsets = [0]
        with open(data_file, 'r', encoding='utf-8') as fp:
            while fp.readline() != '':
                self.offsets.append(fp.tell())
        self.offsets.pop()
        self.fp = open(data_file, 'r', encoding='utf-8')

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.offsets)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        self.fp.seek(self.offsets[idx], 0)
        sample = self.transform(self.fp.readline().strip())
        if self.reader.with_mlm:
            sample = self.reader.create_token_masked_lm_predictions(sample)
        return sample