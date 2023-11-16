"""
Signal processing-based evaluation using waveforms
"""
import numpy as np
import os.path as op
import torchaudio
import tqdm
from tabulate import tabulate
from examples.speech_synthesis.utils import gross_pitch_error, voicing_decision_error, f0_frame_error
from examples.speech_synthesis.evaluation.eval_sp import load_eval_spec

def difference_function(x, n, tau_max):
    if False:
        return 10
    '\n    Compute difference function of data x. This solution is implemented directly\n    with Numpy fft.\n\n\n    :param x: audio data\n    :param n: length of data\n    :param tau_max: integration window size\n    :return: difference function\n    :rtype: list\n    '
    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.0]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min((x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size))
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv

def cumulative_mean_normalized_difference_function(df, n):
    if False:
        return 10
    '\n    Compute cumulative mean normalized difference function (CMND).\n\n    :param df: Difference function\n    :param n: length of data\n    :return: cumulative mean normalized difference function\n    :rtype: list\n    '
    cmn_df = df[1:] * range(1, n) / np.cumsum(df[1:]).astype(float)
    return np.insert(cmn_df, 0, 1)

def get_pitch(cmdf, tau_min, tau_max, harmo_th=0.1):
    if False:
        i = 10
        return i + 15
    '\n    Return fundamental period of a frame based on CMND function.\n\n    :param cmdf: Cumulative Mean Normalized Difference function\n    :param tau_min: minimum period for speech\n    :param tau_max: maximum period for speech\n    :param harmo_th: harmonicity threshold to determine if it is necessary to\n    compute pitch frequency\n    :return: fundamental period if there is values under threshold, 0 otherwise\n    :rtype: float\n    '
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1
    return 0

def compute_yin(sig, sr, w_len=512, w_step=256, f0_min=100, f0_max=500, harmo_thresh=0.1):
    if False:
        return 10
    '\n\n    Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.\n\n    https://github.com/NVIDIA/mellotron adaption of\n    https://github.com/patriceguyot/Yin\n\n    :param sig: Audio signal (list of float)\n    :param sr: sampling rate (int)\n    :param w_len: size of the analysis window (samples)\n    :param w_step: size of the lag between two consecutives windows (samples)\n    :param f0_min: Minimum fundamental frequency that can be detected (hertz)\n    :param f0_max: Maximum fundamental frequency that can be detected (hertz)\n    :param harmo_thresh: Threshold of detection. The yalgorithmù return the\n    first minimum of the CMND function below this threshold.\n\n    :returns:\n\n        * pitches: list of fundamental frequencies,\n        * harmonic_rates: list of harmonic rate values for each fundamental\n        frequency value (= confidence value)\n        * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction\n        * times: list of time of each estimation\n    :rtype: tuple\n    '
    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)
    time_scale = range(0, len(sig) - w_len, w_step)
    times = [t / float(sr) for t in time_scale]
    frames = [sig[t:t + w_len] for t in time_scale]
    pitches = [0.0] * len(time_scale)
    harmonic_rates = [0.0] * len(time_scale)
    argmins = [0.0] * len(time_scale)
    for (i, frame) in enumerate(frames):
        df = difference_function(frame, w_len, tau_max)
        cm_df = cumulative_mean_normalized_difference_function(df, tau_max)
        p = get_pitch(cm_df, tau_min, tau_max, harmo_thresh)
        if np.argmin(cm_df) > tau_min:
            argmins[i] = float(sr / np.argmin(cm_df))
        if p != 0:
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cm_df[p]
        else:
            harmonic_rates[i] = min(cm_df)
    return (pitches, harmonic_rates, argmins, times)

def extract_f0(samples):
    if False:
        for i in range(10):
            print('nop')
    f0_samples = []
    for sample in tqdm.tqdm(samples):
        if not op.isfile(sample['ref']) or not op.isfile(sample['syn']):
            f0_samples.append(None)
            continue
        (yref, sr) = torchaudio.load(sample['ref'])
        (ysyn, _sr) = torchaudio.load(sample['syn'])
        (yref, ysyn) = (yref[0], ysyn[0])
        assert sr == _sr, f'{sr} != {_sr}'
        yref_f0 = compute_yin(yref, sr)
        ysyn_f0 = compute_yin(ysyn, sr)
        f0_samples += [{'ref': yref_f0, 'syn': ysyn_f0}]
    return f0_samples

def eval_f0_error(samples, distortion_fn):
    if False:
        return 10
    results = []
    for sample in tqdm.tqdm(samples):
        if sample is None:
            results.append(None)
            continue
        (yref_f, _, _, yref_t) = sample['ref']
        (ysyn_f, _, _, ysyn_t) = sample['syn']
        yref_f = np.array(yref_f)
        yref_t = np.array(yref_t)
        ysyn_f = np.array(ysyn_f)
        ysyn_t = np.array(ysyn_t)
        distortion = distortion_fn(yref_t, yref_f, ysyn_t, ysyn_f)
        results.append((distortion.item(), len(yref_f), len(ysyn_f)))
    return results

def eval_gross_pitch_error(samples):
    if False:
        i = 10
        return i + 15
    return eval_f0_error(samples, gross_pitch_error)

def eval_voicing_decision_error(samples):
    if False:
        while True:
            i = 10
    return eval_f0_error(samples, voicing_decision_error)

def eval_f0_frame_error(samples):
    if False:
        print('Hello World!')
    return eval_f0_error(samples, f0_frame_error)

def print_results(results, show_bin):
    if False:
        i = 10
        return i + 15
    results = np.array(list(filter(lambda x: x is not None, results)))
    np.set_printoptions(precision=3)

    def _print_result(results):
        if False:
            return 10
        res = {'nutt': len(results), 'error': results[:, 0].mean(), 'std': results[:, 0].std(), 'dur_ref': int(results[:, 1].sum()), 'dur_syn': int(results[:, 2].sum())}
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

def main(eval_f0, gpe, vde, ffe, show_bin):
    if False:
        return 10
    samples = load_eval_spec(eval_f0)
    if gpe or vde or ffe:
        f0_samples = extract_f0(samples)
    if gpe:
        print('===== Evaluate Gross Pitch Error =====')
        results = eval_gross_pitch_error(f0_samples)
        print_results(results, show_bin)
    if vde:
        print('===== Evaluate Voicing Decision Error =====')
        results = eval_voicing_decision_error(f0_samples)
        print_results(results, show_bin)
    if ffe:
        print('===== Evaluate F0 Frame Error =====')
        results = eval_f0_frame_error(f0_samples)
        print_results(results, show_bin)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_f0')
    parser.add_argument('--gpe', action='store_true')
    parser.add_argument('--vde', action='store_true')
    parser.add_argument('--ffe', action='store_true')
    parser.add_argument('--show-bin', action='store_true')
    args = parser.parse_args()
    main(args.eval_f0, args.gpe, args.vde, args.ffe, args.show_bin)