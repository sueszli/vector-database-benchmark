"""
Signal processing-based evaluation using waveforms
"""
import csv
import numpy as np
import os.path as op
import torch
import tqdm
from tabulate import tabulate
import torchaudio
from examples.speech_synthesis.utils import batch_mel_spectral_distortion
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion

def load_eval_spec(path):
    if False:
        for i in range(10):
            print('nop')
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        samples = list(reader)
    return samples

def eval_distortion(samples, distortion_fn, device='cuda'):
    if False:
        while True:
            i = 10
    nmiss = 0
    results = []
    for sample in tqdm.tqdm(samples):
        if not op.isfile(sample['ref']) or not op.isfile(sample['syn']):
            nmiss += 1
            results.append(None)
            continue
        (yref, sr) = torchaudio.load(sample['ref'])
        (ysyn, _sr) = torchaudio.load(sample['syn'])
        (yref, ysyn) = (yref[0].to(device), ysyn[0].to(device))
        assert sr == _sr, f'{sr} != {_sr}'
        (distortion, extra) = distortion_fn([yref], [ysyn], sr, None)[0]
        (_, _, _, _, _, pathmap) = extra
        nins = torch.sum(pathmap.sum(dim=1) - 1)
        ndel = torch.sum(pathmap.sum(dim=0) - 1)
        results.append((distortion.item(), pathmap.size(0), pathmap.size(1), pathmap.sum().item(), nins.item(), ndel.item()))
    return results

def eval_mel_cepstral_distortion(samples, device='cuda'):
    if False:
        while True:
            i = 10
    return eval_distortion(samples, batch_mel_cepstral_distortion, device)

def eval_mel_spectral_distortion(samples, device='cuda'):
    if False:
        while True:
            i = 10
    return eval_distortion(samples, batch_mel_spectral_distortion, device)

def print_results(results, show_bin):
    if False:
        while True:
            i = 10
    results = np.array(list(filter(lambda x: x is not None, results)))
    np.set_printoptions(precision=3)

    def _print_result(results):
        if False:
            i = 10
            return i + 15
        (dist, dur_ref, dur_syn, dur_ali, nins, ndel) = results.sum(axis=0)
        res = {'nutt': len(results), 'dist': dist, 'dur_ref': int(dur_ref), 'dur_syn': int(dur_syn), 'dur_ali': int(dur_ali), 'dist_per_ref_frm': dist / dur_ref, 'dist_per_syn_frm': dist / dur_syn, 'dist_per_ali_frm': dist / dur_ali, 'ins': nins / dur_ref, 'del': ndel / dur_ref}
        print(tabulate([res.values()], res.keys(), floatfmt='.4f'))
    print('>>>> ALL')
    _print_result(results)
    if show_bin:
        edges = [0, 200, 400, 600, 800, 1000, 2000, 4000]
        for i in range(1, len(edges)):
            mask = np.logical_and(results[:, 1] >= edges[i - 1], results[:, 1] < edges[i])
            if not mask.any():
                continue
            bin_results = results[mask]
            print(f'>>>> ({edges[i - 1]}, {edges[i]})')
            _print_result(bin_results)

def main(eval_spec, mcd, msd, show_bin):
    if False:
        return 10
    samples = load_eval_spec(eval_spec)
    device = 'cpu'
    if mcd:
        print('===== Evaluate Mean Cepstral Distortion =====')
        results = eval_mel_cepstral_distortion(samples, device)
        print_results(results, show_bin)
    if msd:
        print('===== Evaluate Mean Spectral Distortion =====')
        results = eval_mel_spectral_distortion(samples, device)
        print_results(results, show_bin)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_spec')
    parser.add_argument('--mcd', action='store_true')
    parser.add_argument('--msd', action='store_true')
    parser.add_argument('--show-bin', action='store_true')
    args = parser.parse_args()
    main(args.eval_spec, args.mcd, args.msd, args.show_bin)