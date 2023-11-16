import os
import subprocess
import threading
from pathlib import Path
import numpy as np
import torch

def fasta_file_path(prefix_path):
    if False:
        print('Hello World!')
    return prefix_path + '.fasta'

class FastaDataset(torch.utils.data.Dataset):
    """
    For loading protein sequence datasets in the common FASTA data format
    """

    def __init__(self, path: str, cache_indices=False):
        if False:
            print('Hello World!')
        self.fn = fasta_file_path(path)
        self.threadlocal = threading.local()
        self.cache = Path(f'{path}.fasta.idx.npy')
        if cache_indices:
            if self.cache.exists():
                (self.offsets, self.sizes) = np.load(self.cache)
            else:
                (self.offsets, self.sizes) = self._build_index(path)
                np.save(self.cache, np.stack([self.offsets, self.sizes]))
        else:
            (self.offsets, self.sizes) = self._build_index(path)

    def _get_file(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self.threadlocal, 'f'):
            self.threadlocal.f = open(self.fn, 'r')
        return self.threadlocal.f

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        f = self._get_file()
        f.seek(self.offsets[idx])
        desc = f.readline().strip()
        line = f.readline()
        seq = ''
        while line != '' and line[0] != '>':
            seq += line.strip()
            line = f.readline()
        return (desc, seq)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.offsets.size

    def _build_index(self, path: str):
        if False:
            print('Hello World!')
        path = fasta_file_path(path)
        bytes_offsets = subprocess.check_output(f"cat {path} | tqdm --bytes --total $(wc -c < {path})| grep --byte-offset '^>' -o | cut -d: -f1", shell=True)
        fasta_lengths = subprocess.check_output(f"""cat {path} | tqdm --bytes --total $(wc -c < {path})| awk '/^>/ {{print "";next;}} {{ printf("%s",$0);}}' | tail -n+2 | awk '{{print length($1)}}'""", shell=True)
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=' ')
        sizes_np = np.fromstring(fasta_lengths, dtype=np.int64, sep=' ')
        return (bytes_np, sizes_np)

    def __setstate__(self, state):
        if False:
            return 10
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        d = {}
        for (i, v) in self.__dict__.items():
            if i != 'threadlocal':
                d[i] = v
        return d

    def __del__(self):
        if False:
            print('Hello World!')
        if hasattr(self.threadlocal, 'f'):
            self.threadlocal.f.close()
            del self.threadlocal.f

    @staticmethod
    def exists(path):
        if False:
            print('Hello World!')
        return os.path.exists(fasta_file_path(path))

class EncodedFastaDataset(FastaDataset):
    """
    The FastaDataset returns raw sequences - this allows us to return
    indices with a dictionary instead.
    """

    def __init__(self, path, dictionary):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(path, cache_indices=True)
        self.dictionary = dictionary

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        (desc, seq) = super().__getitem__(idx)
        return self.dictionary.encode_line(seq, line_tokenizer=list).long()