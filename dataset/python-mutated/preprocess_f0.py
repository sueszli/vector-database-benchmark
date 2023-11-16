import os
import torch
from tqdm import tqdm
from data_utils import load_audio_path
from fairseq.data.codedataset import get_f0_by_filename

def process_one(path, sr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Args:\n        path: audio file path\n        sr: sampling rate\n    '
    try:
        f0 = get_f0_by_filename(path, sr)
    except Exception as e:
        print(f'WARNING: error when processing {path}. set f0 to zero. original error message:\n{e}')
        f0 = None
    return f0

def main(file_path, out_dir, nshards, rank, sampling_rate):
    if False:
        while True:
            i = 10
    audio_paths = load_audio_path(file_path)
    assert nshards <= len(audio_paths) and nshards > 0
    shard_size = len(audio_paths) / nshards
    s = int(round((rank - 1) * shard_size))
    e = int(round(rank * shard_size))
    audio_paths = audio_paths[s:e]
    path_to_f0 = {}
    for (i, audio_path) in enumerate(tqdm(audio_paths)):
        f0 = process_one(audio_path, sampling_rate)
        path_to_f0[audio_path] = f0
    print(f'finished processing {len(path_to_f0)} utterances ({s}-{e})')
    f0_path = f'{out_dir}/f0_{rank}_{nshards}.pt'
    os.makedirs(out_dir, exist_ok=True)
    torch.save(path_to_f0, f0_path)
    print(f'saved to {f0_path}')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    parser.add_argument('out_dir')
    parser.add_argument('--nshards', type=int, default=20)
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--sampling_rate', type=int, default=16000)
    args = parser.parse_args()
    main(**vars(args))