import torch
from fairseq import utils
from . import FairseqDataset

def backtranslate_samples(samples, collate_fn, generate_fn, cuda=True):
    if False:
        i = 10
        return i + 15
    "Backtranslate a list of samples.\n\n    Given an input (*samples*) of the form:\n\n        [{'id': 1, 'source': 'hallo welt'}]\n\n    this will return:\n\n        [{'id': 1, 'source': 'hello world', 'target': 'hallo welt'}]\n\n    Args:\n        samples (List[dict]): samples to backtranslate. Individual samples are\n            expected to have a 'source' key, which will become the 'target'\n            after backtranslation.\n        collate_fn (callable): function to collate samples into a mini-batch\n        generate_fn (callable): function to generate backtranslations\n        cuda (bool): use GPU for generation (default: ``True``)\n\n    Returns:\n        List[dict]: an updated list of samples with a backtranslated source\n    "
    collated_samples = collate_fn(samples)
    s = utils.move_to_cuda(collated_samples) if cuda else collated_samples
    generated_sources = generate_fn(s)
    id_to_src = {sample['id']: sample['source'] for sample in samples}
    return [{'id': id.item(), 'target': id_to_src[id.item()], 'source': hypos[0]['tokens'].cpu()} for (id, hypos) in zip(collated_samples['id'], generated_sources)]

class BacktranslationDataset(FairseqDataset):
    """
    Sets up a backtranslation dataset which takes a tgt batch, generates
    a src using a tgt-src backtranslation function (*backtranslation_fn*),
    and returns the corresponding `{generated src, input tgt}` batch.

    Args:
        tgt_dataset (~fairseq.data.FairseqDataset): the dataset to be
            backtranslated. Only the source side of this dataset will be used.
            After backtranslation, the source sentences in this dataset will be
            returned as the targets.
        src_dict (~fairseq.data.Dictionary): the dictionary of backtranslated
            sentences.
        tgt_dict (~fairseq.data.Dictionary, optional): the dictionary of
            sentences to be backtranslated.
        backtranslation_fn (callable, optional): function to call to generate
            backtranslations. This is typically the `generate` method of a
            :class:`~fairseq.sequence_generator.SequenceGenerator` object.
            Pass in None when it is not available at initialization time, and
            use set_backtranslation_fn function to set it when available.
        output_collater (callable, optional): function to call on the
            backtranslated samples to create the final batch
            (default: ``tgt_dataset.collater``).
        cuda: use GPU for generation
    """

    def __init__(self, tgt_dataset, src_dict, tgt_dict=None, backtranslation_fn=None, output_collater=None, cuda=True, **kwargs):
        if False:
            print('Hello World!')
        self.tgt_dataset = tgt_dataset
        self.backtranslation_fn = backtranslation_fn
        self.output_collater = output_collater if output_collater is not None else tgt_dataset.collater
        self.cuda = cuda if torch.cuda.is_available() else False
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def __getitem__(self, index):
        if False:
            return 10
        '\n        Returns a single sample from *tgt_dataset*. Note that backtranslation is\n        not applied in this step; use :func:`collater` instead to backtranslate\n        a batch of samples.\n        '
        return self.tgt_dataset[index]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.tgt_dataset)

    def set_backtranslation_fn(self, backtranslation_fn):
        if False:
            while True:
                i = 10
        self.backtranslation_fn = backtranslation_fn

    def collater(self, samples):
        if False:
            while True:
                i = 10
        'Merge and backtranslate a list of samples to form a mini-batch.\n\n        Using the samples from *tgt_dataset*, load a collated target sample to\n        feed to the backtranslation model. Then take the backtranslation with\n        the best score as the source and the original input as the target.\n\n        Note: we expect *tgt_dataset* to provide a function `collater()` that\n        will collate samples into the format expected by *backtranslation_fn*.\n        After backtranslation, we will feed the new list of samples (i.e., the\n        `(backtranslated source, original source)` pairs) to *output_collater*\n        and return the result.\n\n        Args:\n            samples (List[dict]): samples to backtranslate and collate\n\n        Returns:\n            dict: a mini-batch with keys coming from *output_collater*\n        '
        if samples[0].get('is_dummy', False):
            return samples
        samples = backtranslate_samples(samples=samples, collate_fn=self.tgt_dataset.collater, generate_fn=lambda net_input: self.backtranslation_fn(net_input), cuda=self.cuda)
        return self.output_collater(samples)

    def num_tokens(self, index):
        if False:
            print('Hello World!')
        'Just use the tgt dataset num_tokens'
        return self.tgt_dataset.num_tokens(index)

    def ordered_indices(self):
        if False:
            return 10
        'Just use the tgt dataset ordered_indices'
        return self.tgt_dataset.ordered_indices()

    def size(self, index):
        if False:
            print('Hello World!')
        "Return an example's size as a float or tuple. This value is used\n        when filtering a dataset with ``--max-positions``.\n\n        Note: we use *tgt_dataset* to approximate the length of the source\n        sentence, since we do not know the actual length until after\n        backtranslation.\n        "
        tgt_size = self.tgt_dataset.size(index)[0]
        return (tgt_size, tgt_size)

    @property
    def supports_prefetch(self):
        if False:
            i = 10
            return i + 15
        return getattr(self.tgt_dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        if False:
            while True:
                i = 10
        return self.tgt_dataset.prefetch(indices)