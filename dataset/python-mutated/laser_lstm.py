import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel, register_model, register_model_architecture

@register_model('laser_lstm')
class LSTMModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        if False:
            while True:
                i = 10
        super().__init__(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None, tgt_tokens=None, tgt_lengths=None, target_language_id=None, dataset_name=''):
        if False:
            i = 10
            return i + 15
        assert target_language_id is not None
        src_encoder_out = self.encoder(src_tokens, src_lengths, dataset_name)
        return self.decoder(prev_output_tokens, src_encoder_out, lang_id=target_language_id)

    @staticmethod
    def add_args(parser):
        if False:
            for i in range(10):
                print('nop')
        'Add model-specific arguments to the parser.'
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D', help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', default=None, type=str, metavar='STR', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N', help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true', help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', default=None, type=str, metavar='STR', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N', help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N', help='decoder output embedding dimension')
        parser.add_argument('--decoder-zero-init', type=str, metavar='BOOL', help='initialize the decoder hidden/cell state to zero')
        parser.add_argument('--decoder-lang-embed-dim', type=int, metavar='N', help='decoder language embedding dimension')
        parser.add_argument('--fixed-embeddings', action='store_true', help='keep embeddings fixed (ENCODER ONLY)')
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D', help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D', help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D', help='dropout probability for decoder output')

    @classmethod
    def build_model(cls, args, task):
        if False:
            return 10
        'Build a new model instance.'
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            if False:
                while True:
                    i = 10
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)
        pretrained_encoder_embed = None
        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(args.decoder_embed_path, task.target_dictionary, args.decoder_embed_dim)
        num_langs = task.num_tasks if hasattr(task, 'num_tasks') else 0
        encoder = LSTMEncoder(dictionary=task.source_dictionary, embed_dim=args.encoder_embed_dim, hidden_size=args.encoder_hidden_size, num_layers=args.encoder_layers, dropout_in=args.encoder_dropout_in, dropout_out=args.encoder_dropout_out, bidirectional=args.encoder_bidirectional, pretrained_embed=pretrained_encoder_embed, fixed_embeddings=args.fixed_embeddings)
        decoder = LSTMDecoder(dictionary=task.target_dictionary, embed_dim=args.decoder_embed_dim, hidden_size=args.decoder_hidden_size, out_embed_dim=args.decoder_out_embed_dim, num_layers=args.decoder_layers, dropout_in=args.decoder_dropout_in, dropout_out=args.decoder_dropout_out, zero_init=options.eval_bool(args.decoder_zero_init), encoder_embed_dim=args.encoder_embed_dim, encoder_output_units=encoder.output_units, pretrained_embed=pretrained_decoder_embed, num_langs=num_langs, lang_embed_dim=args.decoder_lang_embed_dim)
        return cls(encoder, decoder)

class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(self, dictionary, embed_dim=512, hidden_size=512, num_layers=1, dropout_in=0.1, dropout_out=0.1, bidirectional=False, left_pad=True, pretrained_embed=None, padding_value=0.0, fixed_embeddings=False):
        if False:
            print('Hello World!')
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed
        if fixed_embeddings:
            self.embed_tokens.weight.requires_grad = False
        self.lstm = LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=self.dropout_out if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.left_pad = left_pad
        self.padding_value = padding_value
        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths, dataset_name):
        if False:
            while True:
                i = 10
        if self.left_pad:
            src_tokens = utils.convert_padding_direction(src_tokens, self.padding_idx, left_to_right=True)
        (bsz, seqlen) = src_tokens.size()
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = x.transpose(0, 1)
        try:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())
        except BaseException:
            raise Exception(f'Packing failed in dataset {dataset_name}')
        if self.bidirectional:
            state_size = (2 * self.num_layers, bsz, self.hidden_size)
        else:
            state_size = (self.num_layers, bsz, self.hidden_size)
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        (packed_outs, (final_hiddens, final_cells)) = self.lstm(packed_x, (h0, c0))
        (x, _) = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]
        if self.bidirectional:

            def combine_bidir(outs):
                if False:
                    while True:
                        i = 10
                return torch.cat([torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(1, bsz, self.output_units) for i in range(self.num_layers)], dim=0)
            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)
        sentemb = x.max(dim=0)[0]
        return {'sentemb': sentemb, 'encoder_out': (x, final_hiddens, final_cells), 'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None}

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if False:
            print('Hello World!')
        encoder_out_dict['sentemb'] = encoder_out_dict['sentemb'].index_select(0, new_order)
        encoder_out_dict['encoder_out'] = tuple((eo.index_select(1, new_order) for eo in encoder_out_dict['encoder_out']))
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = encoder_out_dict['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out_dict

    def max_positions(self):
        if False:
            print('Hello World!')
        'Maximum input length supported by the encoder.'
        return int(100000.0)

class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512, num_layers=1, dropout_in=0.1, dropout_out=0.1, zero_init=False, encoder_embed_dim=512, encoder_output_units=512, pretrained_embed=None, num_langs=1, lang_embed_dim=0):
        if False:
            while True:
                i = 10
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed
        self.layers = nn.ModuleList([LSTMCell(input_size=encoder_output_units + embed_dim + lang_embed_dim if layer == 0 else hidden_size, hidden_size=hidden_size) for layer in range(num_layers)])
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)
        if zero_init:
            self.sentemb2init = None
        else:
            self.sentemb2init = Linear(encoder_output_units, 2 * num_layers * hidden_size)
        if lang_embed_dim == 0:
            self.embed_lang = None
        else:
            self.embed_lang = nn.Embedding(num_langs, lang_embed_dim)
            nn.init.uniform_(self.embed_lang.weight, -0.1, 0.1)

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None, lang_id=0):
        if False:
            i = 10
            return i + 15
        sentemb = encoder_out_dict['sentemb']
        encoder_out = encoder_out_dict['encoder_out']
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        (bsz, seqlen) = prev_output_tokens.size()
        (encoder_outs, _, _) = encoder_out[:3]
        srclen = encoder_outs.size(0)
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        if self.embed_lang is not None:
            lang_ids = prev_output_tokens.data.new_full((bsz,), lang_id)
            langemb = self.embed_lang(lang_ids)
        x = x.transpose(0, 1)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            (prev_hiddens, prev_cells, input_feed) = cached_state
        else:
            num_layers = len(self.layers)
            if self.sentemb2init is None:
                prev_hiddens = [x.data.new(bsz, self.hidden_size).zero_() for i in range(num_layers)]
                prev_cells = [x.data.new(bsz, self.hidden_size).zero_() for i in range(num_layers)]
            else:
                init = self.sentemb2init(sentemb)
                prev_hiddens = [init[:, 2 * i * self.hidden_size:(2 * i + 1) * self.hidden_size] for i in range(num_layers)]
                prev_cells = [init[:, (2 * i + 1) * self.hidden_size:(2 * i + 2) * self.hidden_size] for i in range(num_layers)]
            input_feed = x.data.new(bsz, self.hidden_size).zero_()
        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            if self.embed_lang is None:
                input = torch.cat((x[j, :, :], sentemb), dim=1)
            else:
                input = torch.cat((x[j, :, :], sentemb, langemb), dim=1)
            for (i, rnn) in enumerate(self.layers):
                (hidden, cell) = rnn(input, (prev_hiddens[i], prev_cells[i]))
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                prev_hiddens[i] = hidden
                prev_cells[i] = cell
            out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)
            input_feed = out
            outs.append(out)
        utils.set_incremental_state(self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        x = x.transpose(1, 0)
        attn_scores = attn_scores.transpose(0, 2)
        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        x = self.fc_out(x)
        return (x, attn_scores)

    def reorder_incremental_state(self, incremental_state, new_order):
        if False:
            print('Hello World!')
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)
        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        if False:
            print('Hello World!')
        'Maximum output length supported by the decoder.'
        return int(100000.0)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    if False:
        while True:
            i = 10
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LSTM(input_size, hidden_size, **kwargs):
    if False:
        while True:
            i = 10
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for (name, param) in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

def LSTMCell(input_size, hidden_size, **kwargs):
    if False:
        while True:
            i = 10
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for (name, param) in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

def Linear(in_features, out_features, bias=True, dropout=0):
    if False:
        for i in range(10):
            print('nop')
    'Weight-normalized Linear layer (input: N x T x C)'
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

@register_model_architecture('laser_lstm', 'laser_lstm')
def base_architecture(args):
    if False:
        print('Hello World!')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.decoder_zero_init = getattr(args, 'decoder_zero_init', '0')
    args.decoder_lang_embed_dim = getattr(args, 'decoder_lang_embed_dim', 0)
    args.fixed_embeddings = getattr(args, 'fixed_embeddings', False)