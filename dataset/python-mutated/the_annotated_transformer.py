import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings('ignore')
RUN_EXAMPLES = True

def is_interactive_notebook():
    if False:
        return 10
    return __name__ == '__main__'

def show_example(fn, args=[]):
    if False:
        i = 10
        return i + 15
    if __name__ == '__main__' and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    if False:
        i = 10
        return i + 15
    if __name__ == '__main__' and RUN_EXAMPLES:
        fn(*args)

class DummyOptimizer(torch.optim.Optimizer):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.param_groups = [{'lr': 0}]
        None

    def step(self):
        if False:
            while True:
                i = 10
        None

    def zero_grad(self, set_to_none=False):
        if False:
            for i in range(10):
                print('nop')
        None

class DummyScheduler:

    def step(self):
        if False:
            return 10
        None

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        if False:
            while True:
                i = 10
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        if False:
            print('Hello World!')
        'Take in and process masked src and target sequences.'
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        if False:
            i = 10
            return i + 15
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        if False:
            return 10
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        if False:
            for i in range(10):
                print('nop')
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        if False:
            print('Hello World!')
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    if False:
        for i in range(10):
            print('nop')
    'Produce N identical layers.'
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        if False:
            return 10
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        if False:
            print('Hello World!')
        'Pass the input (and mask) through each layer in turn.'
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-06):
        if False:
            return 10
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        if False:
            while True:
                i = 10
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        if False:
            for i in range(10):
                print('nop')
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        if False:
            while True:
                i = 10
        'Apply residual connection to any sublayer with the same size.'
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        if False:
            print('Hello World!')
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        if False:
            print('Hello World!')
        'Follow Figure 1 (left) for connections.'
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        if False:
            i = 10
            return i + 15
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        if False:
            while True:
                i = 10
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        if False:
            return 10
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        if False:
            return 10
        'Follow Figure 1 (right) for connections.'
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    if False:
        i = 10
        return i + 15
    'Mask out subsequent positions.'
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def example_mask():
    if False:
        i = 10
        return i + 15
    LS_data = pd.concat([pd.DataFrame({'Subsequent Mask': subsequent_mask(20)[0][x, y].flatten(), 'Window': y, 'Masking': x}) for y in range(20) for x in range(20)])
    return alt.Chart(LS_data).mark_rect().properties(height=250, width=250).encode(alt.X('Window:O'), alt.Y('Masking:O'), alt.Color('Subsequent Mask:Q', scale=alt.Scale(scheme='viridis'))).interactive()
show_example(example_mask)

def attention(query, key, value, mask=None, dropout=None):
    if False:
        for i in range(10):
            print('nop')
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return (torch.matmul(p_attn, value), p_attn)

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        if False:
            return 10
        'Take in model size and number of heads.'
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if False:
            for i in range(10):
                print('nop')
        'Implements Figure 2'
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        (query, key, value) = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for (lin, x) in zip(self.linears, (query, key, value))]
        (x, self.attn) = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        if False:
            return 10
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        if False:
            i = 10
            return i + 15
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        if False:
            return 10
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

def example_positional():
    if False:
        i = 10
        return i + 15
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))
    data = pd.concat([pd.DataFrame({'embedding': y[0, :, dim], 'dimension': dim, 'position': list(range(100))}) for dim in [4, 5, 6, 7]])
    return alt.Chart(data).mark_line().properties(width=800).encode(x='position', y='embedding', color='dimension:N').interactive()
show_example(example_positional)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    if False:
        i = 10
        return i + 15
    'Helper: Construct a model from hyperparameters.'
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N), nn.Sequential(Embeddings(d_model, src_vocab), c(position)), nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), Generator(d_model, tgt_vocab))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def inference_test():
    if False:
        return 10
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)
    for i in range(9):
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = test_model.generator(out[:, -1])
        (_, next_word) = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print('Example Untrained Model Prediction:', ys)

def run_tests():
    if False:
        for i in range(10):
            print('nop')
    for _ in range(10):
        inference_test()
show_example(run_tests)

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):
        if False:
            while True:
                i = 10
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        if False:
            while True:
                i = 10
        'Create a mask to hide padding and future words.'
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0

def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode='train', accum_iter=1, train_state=TrainState()):
    if False:
        return 10
    'Train a single epoch'
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for (i, batch) in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        (loss, loss_node) = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == 'train' or mode == 'train+log':
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == 'train' or mode == 'train+log'):
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start
            print(('Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f ' + '| Tokens / Sec: %7.1f | Learning Rate: %6.1e') % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr))
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return (total_loss / total_tokens, train_state)

def rate(step, model_size, factor, warmup):
    if False:
        i = 10
        return i + 15
    '\n    we have to default the step to 1 for LambdaLR function\n    to avoid zero raising to negative power.\n    '
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

def example_learning_schedule():
    if False:
        print('Hello World!')
    opts = [[512, 1, 4000], [512, 1, 8000], [256, 1, 4000]]
    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []
    for (idx, example) in enumerate(opts):
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-09)
        lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, *example))
        tmp = []
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)
    learning_rates = torch.tensor(learning_rates)
    alt.data_transformers.disable_max_rows()
    opts_data = pd.concat([pd.DataFrame({'Learning Rate': learning_rates[warmup_idx, :], 'model_size:warmup': ['512:4000', '512:8000', '256:4000'][warmup_idx], 'step': range(20000)}) for warmup_idx in [0, 1, 2]])
    return alt.Chart(opts_data).mark_line().properties(width=600).encode(x='step', y='Learning Rate', color='model_size:warmup:N').interactive()
example_learning_schedule()

class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        if False:
            for i in range(10):
                print('nop')
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        if False:
            return 10
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def example_label_smoothing():
    if False:
        while True:
            i = 10
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]])
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat([pd.DataFrame({'target distribution': crit.true_dist[x, y].flatten(), 'columns': y, 'rows': x}) for y in range(5) for x in range(5)])
    return alt.Chart(LS_data).mark_rect(color='Blue', opacity=1).properties(height=200, width=200).encode(alt.X('columns:O', title=None), alt.Y('rows:O', title=None), alt.Color('target distribution:Q', scale=alt.Scale(scheme='viridis'))).interactive()
show_example(example_label_smoothing)

def loss(x, crit):
    if False:
        print('Hello World!')
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data

def penalization_visualization():
    if False:
        i = 10
        return i + 15
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame({'Loss': [loss(x, crit) for x in range(1, 100)], 'Steps': list(range(99))}).astype('float')
    return alt.Chart(loss_data).mark_line().properties(width=350).encode(x='Steps', y='Loss').interactive()
show_example(penalization_visualization)

def data_gen(V, batch_size, nbatches):
    if False:
        for i in range(10):
            print('nop')
    'Generate random data for a src-tgt copy task.'
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion):
        if False:
            for i in range(10):
                print('nop')
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        if False:
            return 10
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return (sloss.data * norm, sloss)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    if False:
        while True:
            i = 10
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        (_, next_word) = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def example_simple_model():
    if False:
        return 10
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-09)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400))
    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(data_gen(V, batch_size, 20), model, SimpleLossCompute(model.generator, criterion), optimizer, lr_scheduler, mode='train')
        model.eval()
        run_epoch(data_gen(V, batch_size, 5), model, SimpleLossCompute(model.generator, criterion), DummyOptimizer(), DummyScheduler(), mode='eval')[0]
    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

def load_tokenizers():
    if False:
        i = 10
        return i + 15
    try:
        spacy_de = spacy.load('de_core_news_sm')
    except IOError:
        os.system('python -m spacy download de_core_news_sm')
        spacy_de = spacy.load('de_core_news_sm')
    try:
        spacy_en = spacy.load('en_core_web_sm')
    except IOError:
        os.system('python -m spacy download en_core_web_sm')
        spacy_en = spacy.load('en_core_web_sm')
    return (spacy_de, spacy_en)

def tokenize(text, tokenizer):
    if False:
        for i in range(10):
            print('nop')
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    if False:
        return 10
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_de, spacy_en):
    if False:
        i = 10
        return i + 15

    def tokenize_de(text):
        if False:
            while True:
                i = 10
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        if False:
            while True:
                i = 10
        return tokenize(text, spacy_en)
    print('Building German Vocabulary ...')
    (train, val, test) = datasets.Multi30k(language_pair=('de', 'en'))
    vocab_src = build_vocab_from_iterator(yield_tokens(train + val + test, tokenize_de, index=0), min_freq=2, specials=['<s>', '</s>', '<blank>', '<unk>'])
    print('Building English Vocabulary ...')
    (train, val, test) = datasets.Multi30k(language_pair=('de', 'en'))
    vocab_tgt = build_vocab_from_iterator(yield_tokens(train + val + test, tokenize_en, index=1), min_freq=2, specials=['<s>', '</s>', '<blank>', '<unk>'])
    vocab_src.set_default_index(vocab_src['<unk>'])
    vocab_tgt.set_default_index(vocab_tgt['<unk>'])
    return (vocab_src, vocab_tgt)

def load_vocab(spacy_de, spacy_en):
    if False:
        for i in range(10):
            print('nop')
    if not exists('vocab.pt'):
        (vocab_src, vocab_tgt) = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), 'vocab.pt')
    else:
        (vocab_src, vocab_tgt) = torch.load('vocab.pt')
    print('Finished.\nVocabulary sizes:')
    print(len(vocab_src))
    print(len(vocab_tgt))
    return (vocab_src, vocab_tgt)
if is_interactive_notebook():
    (spacy_de, spacy_en) = show_example(load_tokenizers)
    (vocab_src, vocab_tgt) = show_example(load_vocab, args=[spacy_de, spacy_en])

def collate_batch(batch, src_pipeline, tgt_pipeline, src_vocab, tgt_vocab, device, max_padding=128, pad_id=2):
    if False:
        i = 10
        return i + 15
    bs_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)
    (src_list, tgt_list) = ([], [])
    for (_src, _tgt) in batch:
        processed_src = torch.cat([bs_id, torch.tensor(src_vocab(src_pipeline(_src)), dtype=torch.int64, device=device), eos_id], 0)
        processed_tgt = torch.cat([bs_id, torch.tensor(tgt_vocab(tgt_pipeline(_tgt)), dtype=torch.int64, device=device), eos_id], 0)
        src_list.append(pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id))
        tgt_list.append(pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id))
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=12000, max_padding=128, is_distributed=True):
    if False:
        return 10

    def tokenize_de(text):
        if False:
            for i in range(10):
                print('nop')
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        if False:
            i = 10
            return i + 15
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        if False:
            print('Hello World!')
        return collate_batch(batch, tokenize_de, tokenize_en, vocab_src, vocab_tgt, device, max_padding=max_padding, pad_id=vocab_src.get_stoi()['<blank>'])
    (train_iter, valid_iter, test_iter) = datasets.Multi30k(language_pair=('de', 'en'))
    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None
    train_dataloader = DataLoader(train_iter_map, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_iter_map, batch_size=batch_size, shuffle=valid_sampler is None, sampler=valid_sampler, collate_fn=collate_fn)
    return (train_dataloader, valid_dataloader)

def train_worker(gpu, ngpus_per_node, vocab_src, vocab_tgt, spacy_de, spacy_en, config, is_distributed=False):
    if False:
        while True:
            i = 10
    print(f'Train worker process using GPU: {gpu} for training', flush=True)
    torch.cuda.set_device(gpu)
    pad_idx = vocab_tgt['<blank>']
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group('nccl', init_method='env://', rank=gpu, world_size=ngpus_per_node)
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)
    (train_dataloader, valid_dataloader) = create_dataloaders(gpu, vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=config['batch_size'] // ngpus_per_node, max_padding=config['max_padding'], is_distributed=is_distributed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], betas=(0.9, 0.98), eps=1e-09)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config['warmup']))
    train_state = TrainState()
    for epoch in range(config['num_epochs']):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)
        model.train()
        print(f'[GPU{gpu}] Epoch {epoch} Training ====', flush=True)
        (_, train_state) = run_epoch((Batch(b[0], b[1], pad_idx) for b in train_dataloader), model, SimpleLossCompute(module.generator, criterion), optimizer, lr_scheduler, mode='train+log', accum_iter=config['accum_iter'], train_state=train_state)
        GPUtil.showUtilization()
        if is_main_process:
            file_path = '%s%.2d.pt' % (config['file_prefix'], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()
        print(f'[GPU{gpu}] Epoch {epoch} Validation ====', flush=True)
        model.eval()
        sloss = run_epoch((Batch(b[0], b[1], pad_idx) for b in valid_dataloader), model, SimpleLossCompute(module.generator, criterion), DummyOptimizer(), DummyScheduler(), mode='eval')
        print(sloss)
        torch.cuda.empty_cache()
    if is_main_process:
        file_path = '%sfinal.pt' % config['file_prefix']
        torch.save(module.state_dict(), file_path)

def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if False:
        for i in range(10):
            print('nop')
    from the_annotated_transformer import train_worker
    ngpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    print(f'Number of GPUs detected: {ngpus}')
    print('Spawning training processes ...')
    mp.spawn(train_worker, nprocs=ngpus, args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True))

def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if False:
        i = 10
        return i + 15
    if config['distributed']:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)

def load_trained_model():
    if False:
        return 10
    config = {'batch_size': 32, 'distributed': False, 'num_epochs': 8, 'accum_iter': 10, 'base_lr': 1.0, 'max_padding': 72, 'warmup': 3000, 'file_prefix': 'multi30k_model_'}
    model_path = 'multi30k_model_final.pt'
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load('multi30k_model_final.pt'))
    return model
if is_interactive_notebook():
    model = load_trained_model()
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight

def average(model, models):
    if False:
        for i in range(10):
            print('nop')
    'Average models into model'
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))

def check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=15, pad_idx=2, eos_string='</s>'):
    if False:
        return 10
    results = [()] * n_examples
    for idx in range(n_examples):
        print('\nExample %d ========\n' % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]
        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]
        print('Source Text (Input)        : ' + ' '.join(src_tokens).replace('\n', ''))
        print('Target Text (Ground Truth) : ' + ' '.join(tgt_tokens).replace('\n', ''))
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = ' '.join([vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]).split(eos_string, 1)[0] + eos_string
        print('Model Output               : ' + model_txt.replace('\n', ''))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results

def run_model_example(n_examples=5):
    if False:
        for i in range(10):
            print('nop')
    global vocab_src, vocab_tgt, spacy_de, spacy_en
    print('Preparing Data ...')
    (_, valid_dataloader) = create_dataloaders(torch.device('cpu'), vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=1, is_distributed=False)
    print('Loading Trained Model ...')
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load('multi30k_model_final.pt', map_location=torch.device('cpu')))
    print('Checking Model Outputs:')
    example_data = check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples)
    return (model, example_data)

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    if False:
        for i in range(10):
            print('nop')
    'convert a dense matrix to a data frame with row and column indices'
    return pd.DataFrame([(r, c, float(m[r, c]), '%.3d %s' % (r, row_tokens[r] if len(row_tokens) > r else '<blank>'), '%.3d %s' % (c, col_tokens[c] if len(col_tokens) > c else '<blank>')) for r in range(m.shape[0]) for c in range(m.shape[1]) if r < max_row and c < max_col], columns=['row', 'column', 'value', 'row_token', 'col_token'])

def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    if False:
        while True:
            i = 10
    df = mtx2df(attn[0, head].data, max_dim, max_dim, row_tokens, col_tokens)
    return alt.Chart(data=df).mark_rect().encode(x=alt.X('col_token', axis=alt.Axis(title='')), y=alt.Y('row_token', axis=alt.Axis(title='')), color='value', tooltip=['row', 'column', 'value', 'row_token', 'col_token']).properties(height=400, width=400).interactive()

def get_encoder(model, layer):
    if False:
        for i in range(10):
            print('nop')
    return model.encoder.layers[layer].self_attn.attn

def get_decoder_self(model, layer):
    if False:
        print('Hello World!')
    return model.decoder.layers[layer].self_attn.attn

def get_decoder_src(model, layer):
    if False:
        for i in range(10):
            print('nop')
    return model.decoder.layers[layer].src_attn.attn

def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    if False:
        return 10
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [attn_map(attn, 0, h, row_tokens=row_tokens, col_tokens=col_tokens, max_dim=ntokens) for h in range(n_heads)]
    assert n_heads == 8
    return alt.vconcat(charts[0] | charts[2] | charts[4] | charts[6]).properties(title='Layer %d' % (layer + 1))

def viz_encoder_self():
    if False:
        for i in range(10):
            print('nop')
    (model, example_data) = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]
    layer_viz = [visualize_layer(model, layer, get_encoder, len(example[1]), example[1], example[1]) for layer in range(6)]
    return alt.hconcat(layer_viz[0] & layer_viz[2] & layer_viz[4])
show_example(viz_encoder_self)

def viz_decoder_self():
    if False:
        print('Hello World!')
    (model, example_data) = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]
    layer_viz = [visualize_layer(model, layer, get_decoder_self, len(example[1]), example[1], example[1]) for layer in range(6)]
    return alt.hconcat(layer_viz[0] & layer_viz[1] & layer_viz[2] & layer_viz[3] & layer_viz[4] & layer_viz[5])
show_example(viz_decoder_self)

def viz_decoder_src():
    if False:
        i = 10
        return i + 15
    (model, example_data) = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]
    layer_viz = [visualize_layer(model, layer, get_decoder_src, max(len(example[1]), len(example[2])), example[1], example[2]) for layer in range(6)]
    return alt.hconcat(layer_viz[0] & layer_viz[1] & layer_viz[2] & layer_viz[3] & layer_viz[4] & layer_viz[5])
show_example(viz_decoder_src)