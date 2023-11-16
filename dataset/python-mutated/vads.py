import argparse
import sys
from copy import deepcopy
from scipy.signal import lfilter
import numpy as np
from tqdm import tqdm
import soundfile as sf
import os.path as osp

def get_parser():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='compute vad segments')
    parser.add_argument('--rvad-home', '-r', help='path to rvad home (see https://github.com/zhenghuatan/rVADfast)', required=True)
    return parser

def rvad(speechproc, path):
    if False:
        for i in range(10):
            print('nop')
    (winlen, ovrlen, pre_coef, nfilter, nftt) = (0.025, 0.01, 0.97, 20, 512)
    ftThres = 0.5
    vadThres = 0.4
    opts = 1
    (data, fs) = sf.read(path)
    assert fs == 16000, 'sample rate must be 16khz'
    (ft, flen, fsh10, nfr10) = speechproc.sflux(data, fs, winlen, ovrlen, nftt)
    pv01 = np.zeros(ft.shape[0])
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)
    pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.977, -0.977])
    a = np.array([1.0, -0.954])
    fdata = lfilter(b, a, data, axis=0)
    (noise_samp, noise_seg, n_noise_samp) = speechproc.snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0
    vad_seg = speechproc.snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)
    return (vad_seg, data)

def main():
    if False:
        print('Hello World!')
    parser = get_parser()
    args = parser.parse_args()
    sys.path.append(args.rvad_home)
    import speechproc
    stride = 160
    lines = sys.stdin.readlines()
    root = lines[0].rstrip()
    for fpath in tqdm(lines[1:]):
        path = osp.join(root, fpath.split()[0])
        (vads, wav) = rvad(speechproc, path)
        start = None
        vad_segs = []
        for (i, v) in enumerate(vads):
            if start is None and v == 1:
                start = i * stride
            elif start is not None and v == 0:
                vad_segs.append((start, i * stride))
                start = None
        if start is not None:
            vad_segs.append((start, len(wav)))
        print(' '.join((f'{v[0]}:{v[1]}' for v in vad_segs)))
if __name__ == '__main__':
    main()