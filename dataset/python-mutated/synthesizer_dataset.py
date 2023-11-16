import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence

class SynthesizerDataset(Dataset):

    def __init__(self, metadata_fpath: Path, mel_dir: Path, embed_dir: Path, hparams):
        if False:
            for i in range(10):
                print('nop')
        print('Using inputs from:\n\t%s\n\t%s\n\t%s' % (metadata_fpath, mel_dir, embed_dir))
        with metadata_fpath.open('r') as metadata_file:
            metadata = [line.split('|') for line in metadata_file]
        mel_fnames = [x[1] for x in metadata if int(x[4])]
        mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        embed_fnames = [x[2] for x in metadata if int(x[4])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        self.samples_fpaths = list(zip(mel_fpaths, embed_fpaths))
        self.samples_texts = [x[5].strip() for x in metadata if int(x[4])]
        self.metadata = metadata
        self.hparams = hparams
        print('Found %d samples' % len(self.samples_fpaths))

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if index is list:
            index = index[0]
        (mel_path, embed_path) = self.samples_fpaths[index]
        mel = np.load(mel_path).T.astype(np.float32)
        embed = np.load(embed_path)
        text = text_to_sequence(self.samples_texts[index], self.hparams.tts_cleaner_names)
        text = np.asarray(text).astype(np.int32)
        return (text, mel.astype(np.float32), embed.astype(np.float32), index)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.samples_fpaths)

def collate_synthesizer(batch, r, hparams):
    if False:
        for i in range(10):
            print('nop')
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)
    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r
    if hparams.symmetric_mels:
        mel_pad_value = -1 * hparams.max_abs_value
    else:
        mel_pad_value = 0
    mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = np.stack(mel)
    embeds = np.array([x[2] for x in batch])
    indices = [x[3] for x in batch]
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    embeds = torch.tensor(embeds)
    return (chars, mel, embeds, indices)

def pad1d(x, max_len, pad_value=0):
    if False:
        return 10
    return np.pad(x, (0, max_len - len(x)), mode='constant', constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    if False:
        while True:
            i = 10
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant', constant_values=pad_value)