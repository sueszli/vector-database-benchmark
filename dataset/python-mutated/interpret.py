from __future__ import annotations
from .data.all import *
from .optimizer import *
from .learner import *
from .tabular.core import *
import sklearn.metrics as skm
__all__ = ['plot_top_losses', 'Interpretation', 'ClassificationInterpretation', 'SegmentationInterpretation']

@typedispatch
def plot_top_losses(x, y, *args, **kwargs):
    if False:
        while True:
            i = 10
    raise Exception(f'plot_top_losses is not implemented for {type(x)},{type(y)}')
_all_ = ['plot_top_losses']

class Interpretation:
    """Interpretation base class, can be inherited for task specific Interpretation classes"""

    def __init__(self, learn: Learner, dl: DataLoader, losses: TensorBase, act=None):
        if False:
            for i in range(10):
                print('nop')
        store_attr()

    def __getitem__(self, idxs):
        if False:
            for i in range(10):
                print('nop')
        'Return inputs, preds, targs, decoded outputs, and losses at `idxs`'
        if isinstance(idxs, Tensor):
            idxs = idxs.tolist()
        if not is_listy(idxs):
            idxs = [idxs]
        items = getattr(self.dl.items, 'iloc', L(self.dl.items))[idxs]
        tmp_dl = self.learn.dls.test_dl(items, with_labels=True, process=not isinstance(self.dl, TabDataLoader))
        (inps, preds, targs, decoded) = self.learn.get_preds(dl=tmp_dl, with_input=True, with_loss=False, with_decoded=True, act=self.act, reorder=False)
        return (inps, preds, targs, decoded, self.losses[idxs])

    @classmethod
    def from_learner(cls, learn, ds_idx: int=1, dl: DataLoader=None, act=None):
        if False:
            for i in range(10):
                print('nop')
        'Construct interpretation object from a learner'
        if dl is None:
            dl = learn.dls[ds_idx].new(shuffle=False, drop_last=False)
        (_, _, losses) = learn.get_preds(dl=dl, with_input=False, with_loss=True, with_decoded=False, with_preds=False, with_targs=False, act=act)
        return cls(learn, dl, losses, act)

    def top_losses(self, k: int | None=None, largest: bool=True, items: bool=False):
        if False:
            while True:
                i = 10
        '`k` largest(/smallest) losses and indexes, defaulting to all losses.'
        (losses, idx) = self.losses.topk(ifnone(k, len(self.losses)), largest=largest)
        if items:
            return (losses, idx, getattr(self.dl.items, 'iloc', L(self.dl.items))[idx])
        else:
            return (losses, idx)

    def plot_top_losses(self, k: int | MutableSequence, largest: bool=True, **kwargs):
        if False:
            i = 10
            return i + 15
        'Show `k` largest(/smallest) preds and losses. Implementation based on type dispatch'
        if is_listy(k) or isinstance(k, range):
            (losses, idx) = (o[k] for o in self.top_losses(None, largest))
        else:
            (losses, idx) = self.top_losses(k, largest)
        (inps, preds, targs, decoded, _) = self[idx]
        (inps, targs, decoded) = (tuplify(inps), tuplify(targs), tuplify(decoded))
        (x, y, its) = self.dl._pre_show_batch(inps + targs, max_n=len(idx))
        (x1, y1, outs) = self.dl._pre_show_batch(inps + decoded, max_n=len(idx))
        if its is not None:
            plot_top_losses(x, y, its, outs.itemgot(slice(len(inps), None)), preds, losses, **kwargs)

    def show_results(self, idxs: list, **kwargs):
        if False:
            return 10
        'Show predictions and targets of `idxs`'
        if isinstance(idxs, Tensor):
            idxs = idxs.tolist()
        if not is_listy(idxs):
            idxs = [idxs]
        (inps, _, targs, decoded, _) = self[idxs]
        b = tuplify(inps) + tuplify(targs)
        self.dl.show_results(b, tuplify(decoded), max_n=len(idxs), **kwargs)

class ClassificationInterpretation(Interpretation):
    """Interpretation methods for classification models."""

    def __init__(self, learn: Learner, dl: DataLoader, losses: TensorBase, act=None):
        if False:
            print('Hello World!')
        super().__init__(learn, dl, losses, act)
        self.vocab = self.dl.vocab
        if is_listy(self.vocab):
            self.vocab = self.vocab[-1]

    def confusion_matrix(self):
        if False:
            print('Hello World!')
        'Confusion matrix as an `np.ndarray`.'
        x = torch.arange(0, len(self.vocab))
        (_, targs, decoded) = self.learn.get_preds(dl=self.dl, with_decoded=True, with_preds=True, with_targs=True, act=self.act)
        (d, t) = flatten_check(decoded, targs)
        cm = ((d == x[:, None]) & (t == x[:, None, None])).long().sum(2)
        return to_np(cm)

    def plot_confusion_matrix(self, normalize: bool=False, title: str='Confusion matrix', cmap: str='Blues', norm_dec: int=2, plot_txt: bool=True, **kwargs):
        if False:
            print('Hello World!')
        'Plot the confusion matrix, with `title` and using `cmap`.'
        cm = self.confusion_matrix()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(self.vocab))
        plt.xticks(tick_marks, self.vocab, rotation=90)
        plt.yticks(tick_marks, self.vocab, rotation=0)
        if plot_txt:
            thresh = cm.max() / 2.0
            for (i, j) in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment='center', verticalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        ax = fig.gca()
        ax.set_ylim(len(self.vocab) - 0.5, -0.5)
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)

    def most_confused(self, min_val=1):
        if False:
            i = 10
            return i + 15
        'Sorted descending largest non-diagonal entries of confusion matrix (actual, predicted, # occurrences'
        cm = self.confusion_matrix()
        np.fill_diagonal(cm, 0)
        res = [(self.vocab[i], self.vocab[j], cm[i, j]) for (i, j) in zip(*np.where(cm >= min_val))]
        return sorted(res, key=itemgetter(2), reverse=True)

    def print_classification_report(self):
        if False:
            print('Hello World!')
        'Print scikit-learn classification report'
        (_, targs, decoded) = self.learn.get_preds(dl=self.dl, with_decoded=True, with_preds=True, with_targs=True, act=self.act)
        (d, t) = flatten_check(decoded, targs)
        names = [str(v) for v in self.vocab]
        print(skm.classification_report(t, d, labels=list(self.vocab.o2i.values()), target_names=names))

class SegmentationInterpretation(Interpretation):
    """Interpretation methods for segmentation models."""
    pass