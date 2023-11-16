import gzip
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):

    def __init__(self, filename='./data/shakespeare.txt.gz'):
        if False:
            return 10
        self.len = 0
        with gzip.open(filename, 'rt') as f:
            self.targetLines = [x.strip() for x in f if x.strip()]
            self.srcLines = [x.lower().replace(' ', '') for x in self.targetLines]
            self.len = len(self.srcLines)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return (self.srcLines[index], self.targetLines[index])

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.len
if __name__ == '__main__':
    dataset = TextDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True, num_workers=2)
    for (i, (src, target)) in enumerate(train_loader):
        print(i, 'data', src)