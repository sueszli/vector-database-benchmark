import os
from collections import OrderedDict
from typing import Any, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.sv.DTDNN import CAMPPlus
from modelscope.utils.constant import Tasks
from modelscope.utils.device import create_device

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, n_units, h=8, dropout=0.1):
        if False:
            while True:
                i = 10
        super(MultiHeadSelfAttention, self).__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.att = None

    def forward(self, x, batch_size):
        if False:
            print('Hello World!')
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1)) / np.sqrt(self.d_k)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        x = torch.matmul(p_att, v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, self.h * self.d_k)
        return self.linearO(x)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, n_units, d_units, dropout):
        if False:
            i = 10
            return i + 15
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(n_units, d_units)
        self.linear2 = nn.Linear(d_units, n_units)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PosEncoding(nn.Module):

    def __init__(self, max_seq_len, d_word_vec):
        if False:
            for i in range(10):
                print('nop')
        super(PosEncoding, self).__init__()
        pos_enc = np.array([[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)] for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)
        self.pos_enc = torch.nn.Embedding(max_seq_len + 1, d_word_vec)
        self.pos_enc.weight = torch.nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_len):
        if False:
            return 10
        max_len = torch.max(input_len)
        input_pos = torch.LongTensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        input_pos = input_pos.to(list(self.pos_enc.parameters())[0].device)
        return self.pos_enc(input_pos)

class TransformerEncoder(nn.Module):

    def __init__(self, idim, n_units=256, n_layers=2, e_units=512, h=4, dropout=0.1):
        if False:
            return 10
        super(TransformerEncoder, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format('lnorm1_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('self_att_', i), MultiHeadSelfAttention(n_units, h))
            setattr(self, '{}{:d}'.format('lnorm2_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('ff_', i), PositionwiseFeedForward(n_units, e_units, dropout))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        (bs, num, tframe, dim) = x.size()
        x = x.reshape(bs * num, tframe, -1)
        (B_size, T_size, _) = x.shape
        e = self.linear_in(x.reshape(B_size * T_size, -1))
        for i in range(self.n_layers):
            e = getattr(self, '{}{:d}'.format('lnorm1_', i))(e)
            s = getattr(self, '{}{:d}'.format('self_att_', i))(e, x.shape[0])
            e = e + self.dropout(s)
            e = getattr(self, '{}{:d}'.format('lnorm2_', i))(e)
            s = getattr(self, '{}{:d}'.format('ff_', i))(e)
            e = e + self.dropout(s)
        output = self.lnorm_out(e).reshape(B_size, T_size, -1)
        output = output.reshape(bs, num, tframe, -1)
        return output

class TransformerEncoder_out(nn.Module):

    def __init__(self, idim, n_units=256, n_layers=2, e_units=512, h=4, dropout=0.1):
        if False:
            return 10
        super(TransformerEncoder_out, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format('lnorm1_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('self_att_', i), MultiHeadSelfAttention(n_units, h))
            setattr(self, '{}{:d}'.format('lnorm2_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('ff_', i), PositionwiseFeedForward(n_units, e_units, dropout))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):
        if False:
            while True:
                i = 10
        (B_size, T_size, _) = x.shape
        e = self.linear_in(x.reshape(B_size * T_size, -1))
        for i in range(self.n_layers):
            e = getattr(self, '{}{:d}'.format('lnorm1_', i))(e)
            s = getattr(self, '{}{:d}'.format('self_att_', i))(e, x.shape[0])
            e = e + self.dropout(s)
            e = getattr(self, '{}{:d}'.format('lnorm2_', i))(e)
            s = getattr(self, '{}{:d}'.format('ff_', i))(e)
            e = e + self.dropout(s)
        output = self.lnorm_out(e).reshape(B_size, T_size, -1)
        return output

class OutLayer(nn.Module):

    def __init__(self, n_units=256, num_anchors=2):
        if False:
            print('Hello World!')
        super(OutLayer, self).__init__()
        self.combine = TransformerEncoder_out(num_anchors * n_units, n_units)
        self.out_linear = nn.Linear(n_units // num_anchors, 1)

    def forward(self, input):
        if False:
            while True:
                i = 10
        (bs, num, tframe, dim) = input.size()
        output = input.permute(0, 2, 1, 3).reshape(bs, tframe, -1)
        output = self.combine(output)
        output = output.reshape(bs, tframe, num, -1)
        output = self.out_linear(output).squeeze(-1)
        return output

class TransformerDetector(nn.Module):

    def __init__(self, frame_dim=512, anchor_dim=192, hidden_dim=256, max_seq_len=1000):
        if False:
            return 10
        super(TransformerDetector, self).__init__()
        self.detection = TransformerEncoder(idim=frame_dim + anchor_dim, n_units=hidden_dim)
        self.output = OutLayer(n_units=hidden_dim)
        self.pos_enc = PosEncoding(max_seq_len, hidden_dim)

    def forward(self, feats, anchors):
        if False:
            while True:
                i = 10
        num_frames = feats.shape[1]
        num_anchors = anchors.shape[1]
        bs = feats.shape[0]
        feats = feats.unsqueeze(1).repeat(1, num_anchors, 1, 1)
        anchors = anchors.unsqueeze(2).repeat(1, 1, num_frames, 1)
        sd_in = torch.cat((feats, anchors), dim=-1)
        sd_out = self.detection(sd_in)
        pos_emb = self.pos_enc(torch.tensor([num_frames] * (bs * num_anchors)))
        pos_emb = pos_emb.reshape(bs, num_anchors, num_frames, -1)
        sd_out += pos_emb
        output = self.output(sd_out)
        return output

@MODELS.register_module(Tasks.speaker_diarization, module_name=Models.scl_sd)
class SpeakerChangeLocatorTransformer(TorchModel):
    """A speaekr change locator using the transformer architecture as the backbone.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args, **kwargs):
        if False:
            return 10
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config
        self.feature_dim = self.model_config['fbank_dim']
        frame_size = self.model_config['frame_size']
        anchor_size = self.model_config['anchor_size']
        self.device = create_device(kwargs['device'])
        self.encoder = CAMPPlus(self.feature_dim, output_level='frame')
        self.backend = TransformerDetector(frame_dim=frame_size, anchor_dim=anchor_size)
        pretrained_encoder = kwargs['pretrained_encoder']
        pretrained_backend = kwargs['pretrained_backend']
        self.__load_check_point(pretrained_encoder, pretrained_backend)
        self.encoder.to(self.device)
        self.backend.to(self.device)
        self.encoder.eval()
        self.backend.eval()

    def forward(self, audio, anchors):
        if False:
            while True:
                i = 10
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        assert len(audio.shape) == 2 and audio.shape[0] == 1, 'modelscope error: the shape of input audio to model needs to be [1, T]'
        assert len(anchors.shape) == 3 and anchors.shape[0] == 1 and (anchors.shape[1] == 2), 'modelscope error: the shape of input anchors to model needs to be [1, 2, D]'
        feature = self.__extract_feature(audio)
        frame_state = self.encoder(feature.to(self.device))
        output = self.backend(frame_state, anchors.to(self.device))
        output = output.squeeze(0).detach().cpu().sigmoid()
        time_scale_factor = int(np.ceil(feature.shape[1] / output.shape[0]))
        output = output.unsqueeze(1).expand(-1, time_scale_factor, -1).reshape(-1, output.shape[-1])
        return output

    def __extract_feature(self, audio):
        if False:
            return 10
        feature = Kaldi.fbank(audio, num_mel_bins=self.feature_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature.unsqueeze(0)
        return feature

    def __load_check_point(self, pretrained_encoder, pretrained_backend):
        if False:
            print('Hello World!')
        self.encoder.load_state_dict(torch.load(os.path.join(self.model_dir, pretrained_encoder), map_location=torch.device('cpu')))
        self.backend.load_state_dict(torch.load(os.path.join(self.model_dir, pretrained_backend), map_location=torch.device('cpu')))