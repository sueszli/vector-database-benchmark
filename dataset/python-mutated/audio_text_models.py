import torch
import torchaudio_models as models
from torch import nn, Tensor
from utils import check_for_functorch, extract_weights, GetterReturnType, load_weights
has_functorch = check_for_functorch()

def get_wav2letter(device: torch.device) -> GetterReturnType:
    if False:
        return 10
    N = 10
    input_frames = 700
    vocab_size = 28
    model = models.Wav2Letter(num_classes=vocab_size)
    criterion = torch.nn.NLLLoss()
    model.to(device)
    (params, names) = extract_weights(model)
    inputs = torch.rand([N, 1, input_frames], device=device)
    labels = torch.rand(N, 3, device=device).mul(vocab_size).long()

    def forward(*new_params: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        load_weights(model, names, new_params)
        out = model(inputs)
        loss = criterion(out, labels)
        return loss
    return (forward, params)

def get_deepspeech(device: torch.device) -> GetterReturnType:
    if False:
        while True:
            i = 10
    sample_rate = 16000
    window_size = 0.02
    window = 'hamming'
    audio_conf = dict(sample_rate=sample_rate, window_size=window_size, window=window, noise_dir=None)
    N = 10
    num_classes = 10
    spectrogram_size = 161
    seq_length = 500
    target_length = 10
    labels = torch.rand(num_classes, device=device)
    inputs = torch.rand(N, 1, spectrogram_size, seq_length, device=device)
    inputs_sizes = torch.rand(N, device=device).mul(seq_length * 0.1).add(seq_length * 0.8)
    targets = torch.rand(N, target_length, device=device)
    targets_sizes = torch.full((N,), target_length, dtype=torch.int, device=device)
    model = models.DeepSpeech(rnn_type=nn.LSTM, labels=labels, rnn_hidden_size=1024, nb_layers=5, audio_conf=audio_conf, bidirectional=True)
    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_
        replace_all_batch_norm_modules_(model)
    model = model.to(device)
    criterion = nn.CTCLoss()
    (params, names) = extract_weights(model)

    def forward(*new_params: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        load_weights(model, names, new_params)
        (out, out_sizes) = model(inputs, inputs_sizes)
        out = out.transpose(0, 1)
        loss = criterion(out, targets, out_sizes, targets_sizes)
        return loss
    return (forward, params)

def get_transformer(device: torch.device) -> GetterReturnType:
    if False:
        for i in range(10):
            print('nop')
    N = 64
    seq_length = 128
    ntoken = 50
    model = models.TransformerModel(ntoken=ntoken, ninp=720, nhead=12, nhid=2048, nlayers=2)
    model.to(device)
    if has_functorch:
        model.eval()
    criterion = nn.NLLLoss()
    (params, names) = extract_weights(model)
    data = torch.rand(N, seq_length + 1, device=device).mul(ntoken).long()
    inputs = data.narrow(1, 0, seq_length)
    targets = data.narrow(1, 1, seq_length)

    def forward(*new_params: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        load_weights(model, names, new_params)
        out = model(inputs)
        loss = criterion(out.reshape(N * seq_length, ntoken), targets.reshape(N * seq_length))
        return loss
    return (forward, params)

def get_multiheadattn(device: torch.device) -> GetterReturnType:
    if False:
        print('Hello World!')
    (embed_dim, nhead, tgt_len, src_len, bsz) = (10, 5, 6, 10, 64)
    in_proj = models.InProjContainer(torch.nn.Linear(embed_dim, embed_dim, bias=False), torch.nn.Linear(embed_dim, embed_dim, bias=False), torch.nn.Linear(embed_dim, embed_dim, bias=False))
    model = models.MultiheadAttentionContainer(nhead, in_proj, models.ScaledDotProduct(), torch.nn.Linear(embed_dim, embed_dim, bias=False))
    model.to(device)
    (params, names) = extract_weights(model)
    query = torch.rand((tgt_len, bsz, embed_dim), device=device)
    key = value = torch.rand((src_len, bsz, embed_dim), device=device)
    attn_mask_2D = torch.randint(0, 2, (tgt_len, src_len), device=device).to(torch.bool)
    bias_k = bias_v = torch.rand((1, 1, embed_dim), device=device)
    attn_mask = torch.stack([attn_mask_2D] * (bsz * nhead))
    bias_k = bias_k.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1)
    bias_v = bias_v.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1)

    def forward(*new_params: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        load_weights(model, names, new_params)
        (mha_output, attn_weights) = model(query, key, value, attn_mask=attn_mask, bias_k=bias_k, bias_v=bias_v)
        loss = mha_output.sum() + attn_weights.sum()
        return loss
    return (forward, params)