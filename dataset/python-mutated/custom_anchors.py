import random
import torch
import numpy as np
from scipy.cluster.vq import kmeans

def generate_anchors(dataset, n=9, img_size=416, thr=4.0, gen=1000, verbose=True):
    if False:
        print('Hello World!')
    ' Creates kmeans-evolved anchors from training dataset\n\n        Arguments:\n            dataset: a loaded dataset i.e. subclass of torch.utils.data.Dataset\n            n: number of anchors\n            img_size: image size used for training\n            thr: anchor-label wh ratio threshold used for training, default=4.0\n            gen: generations to evolve anchors using genetic algorithm\n            verbose: print all results\n\n        Return:\n            k: kmeans evolved anchors\n    '
    thr = 1 / thr

    def metric(k, wh):
        if False:
            while True:
                i = 10
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        return (x, x.max(1)[0])

    def anchor_fitness(k):
        if False:
            i = 10
            return i + 15
        (_, best) = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

    def print_results(k, verbose=True):
        if False:
            print('Hello World!')
        k = k[np.argsort(k.prod(1))]
        if verbose:
            (x, best) = metric(k, wh0)
            (bpr, aat) = ((best > thr).float().mean(), (x > thr).float().mean() * n)
            s = f'thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\nn={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, past_thr={x[x > thr].mean():.3f}-mean: '
            print(s)
        return k
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for (s, l) in zip(shapes, dataset.labels)])
    i = (wh0 < 3.0).any(1).sum()
    if i and verbose:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]
    s = wh.std(0)
    (k, dist) = kmeans(wh / s, n, iter=30)
    assert len(k) == n, f'ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)
    wh0 = torch.tensor(wh0, dtype=torch.float32)
    k = print_results(k, verbose=False)
    npr = np.random
    (f, sh, mp, s) = (anchor_fitness(k), k.shape, 0.9, 0.1)
    if verbose:
        print('Generating anchor boxes for training images...')
    for _ in range(gen):
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            (f, k) = (fg, kg.copy())
    return print_results(k)