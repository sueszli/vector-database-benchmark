"""Implementation of the RNN based DrQA reader."""
import torch
import torch.nn as nn
from . import layers

class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        if False:
            i = 10
            return i + 15
        super(RnnDocReader, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim
        self.doc_rnn = layers.StackedBRNN(input_size=doc_input_size, hidden_size=args.hidden_size, num_layers=args.doc_layers, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=args.concat_rnn_layers, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        self.question_rnn = layers.StackedBRNN(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.question_layers, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=args.concat_rnn_layers, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)
        self.start_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size, normalize=normalize)
        self.end_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size, normalize=normalize)

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        if False:
            print('Hello World!')
        'Inputs:\n        x1 = document word indices             [batch * len_d]\n        x1_f = document word features indices  [batch * len_d * nfeat]\n        x1_mask = document padding mask        [batch * len_d]\n        x2 = question word indices             [batch * len_q]\n        x2_mask = question padding mask        [batch * len_q]\n        '
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
        drnn_input = [x1_emb]
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)
        if self.args.num_features > 0:
            drnn_input.append(x1_f)
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return (start_scores, end_scores)