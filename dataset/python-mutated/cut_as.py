import torchaudio
import argparse
import json
import pathlib

def get_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser('Assuring generated audio have the same length as ground-truth audio')
    parser.add_argument('--samples_dir', required=True, type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--prompts_description', required=True, type=str)
    return parser.parse_args()

def cut(src, tgt, l):
    if False:
        while True:
            i = 10
    (x, sr) = torchaudio.load(str(src))
    assert sr == 16000
    x = x.squeeze()
    target_frames = int(l * sr)
    flag = 0
    if target_frames <= x.size(0):
        x = x[:target_frames]
        flag = 1
    else:
        flag = 0
    torchaudio.save(str(tgt), x.unsqueeze(0), sr)
    return flag

def main():
    if False:
        i = 10
        return i + 15
    args = get_args()
    tgt_dir = pathlib.Path(args.out_dir)
    tgt_dir.mkdir(exist_ok=True, parents=True)
    (total_files, sufficiently_long) = (0, 0)
    with open(args.prompts_description, 'r') as f:
        description = json.loads(f.read())
    for src_f in pathlib.Path(args.samples_dir).glob('*.wav'):
        name_prompt = src_f.with_suffix('').name.split('__')[0]
        assert name_prompt in description, f'Cannot find {name_prompt}!'
        target_length = description[name_prompt][0]
        tgt_f = tgt_dir / src_f.name
        is_long_enough = cut(src_f, tgt_f, target_length)
        sufficiently_long += is_long_enough
        if not is_long_enough:
            print(f'{src_f} is not long enough')
        total_files += 1
    print(f'Total files: {total_files}; sufficiently long: {sufficiently_long}')
if __name__ == '__main__':
    main()