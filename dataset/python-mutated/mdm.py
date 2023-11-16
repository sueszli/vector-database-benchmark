import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotation2xyz import Rotation2xyz

class MDM(nn.Module):

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, smpl_data_path=None, ablation=None, activation='gelu', legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512, arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.0)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        if self.arch == 'trans_enc':
            print('TRANS_ENC init')
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print('TRANS_DEC init')
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
        elif self.arch == 'gru':
            print('GRU init')
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)
        self.rot2xyz = Rotation2xyz(device='cpu', smpl_data_path=smpl_data_path, dataset=self.dataset)

    def parameters_wo_clip(self):
        if False:
            return 10
        return [p for (name, p) in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        if False:
            print('Hello World!')
        (clip_model, clip_preprocess) = clip.load(clip_version, device='cpu', jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        if False:
            return 10
        (bs, d) = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        if False:
            return 10
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)
            zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device)
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper\n        timesteps: [batch_size] (int)\n        '
        (bs, njoints, nfeats, nframes) = x.shape
        emb = self.embed_timestep(timesteps)
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)
            emb_gru = emb_gru.permute(1, 2, 0)
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)
            x = torch.cat((x_reshaped, emb_gru), axis=1)
        x = self.input_process(x)
        if self.arch == 'trans_enc':
            xseq = torch.cat((emb, x), axis=0)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)[1:]
        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:]
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)
            (output, _) = self.gru(xseq)
        output = self.output_process(output)
        return output

    def _apply(self, fn):
        if False:
            i = 10
            return i + 15
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        if False:
            while True:
                i = 10
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if False:
            return 10
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):

    def __init__(self, latent_dim, sequence_pos_encoder):
        if False:
            return 10
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(nn.Linear(self.latent_dim, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))

    def forward(self, timesteps):
        if False:
            print('Hello World!')
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class InputProcess(nn.Module):

    def __init__(self, data_rep, input_feats, latent_dim):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        if False:
            print('Hello World!')
        (bs, njoints, nfeats, nframes) = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]
            first_pose = self.poseEmbedding(first_pose)
            vel = x[1:]
            vel = self.velEmbedding(vel)
            return torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError

class OutputProcess(nn.Module):

    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        if False:
            return 10
        (nframes, bs, d) = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]
            first_pose = self.poseFinal(first_pose)
            vel = output[1:]
            vel = self.velFinal(vel)
            output = torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)
        return output

class EmbedAction(nn.Module):

    def __init__(self, num_actions, latent_dim):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        if False:
            print('Hello World!')
        idx = input[:, 0].to(torch.long)
        output = self.action_embedding[idx]
        return output