"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill
which is fixed in the following code
"""
import math
from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import SequenceType

class MaskedCausalAttention(nn.Module):
    """
    Overview:
        The implementation of masked causal attention in decision transformer. The input of this module is a sequence         of several tokens. For the calculated hidden embedding for the i-th token, it is only related the 0 to i-1         input tokens by applying a mask to the attention map. Thus, this module is called masked-causal attention.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Initialize the MaskedCausalAttention Model according to input arguments.\n        Arguments:\n            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.\n            - max_T (:obj:`int`): The max context length of the attention, such as 6.\n            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.\n            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.\n        '
        super().__init__()
        self.n_heads = n_heads
        self.max_T = max_T
        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        self.proj_net = nn.Linear(h_dim, h_dim)
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            MaskedCausalAttention forward computation graph, input a sequence tensor             and return a tensor with the same shape.\n        Arguments:\n            - x (:obj:`torch.Tensor`): The input tensor.\n        Returns:\n            - out (:obj:`torch.Tensor`): Output tensor, the shape is the same as the input.\n        Examples:\n            >>> inputs = torch.randn(2, 4, 64)\n            >>> model = MaskedCausalAttention(64, 5, 4, 0.1)\n            >>> outputs = model(inputs)\n            >>> assert outputs.shape == torch.Size([2, 4, 64])\n        '
        (B, T, C) = x.shape
        (N, D) = (self.n_heads, C // self.n_heads)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        normalized_weights = F.softmax(weights, dim=-1)
        attention = self.att_drop(normalized_weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        out = self.proj_drop(self.proj_net(attention))
        return out

class Block(nn.Module):
    """
    Overview:
        The implementation of a transformer block in decision transformer.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float) -> None:
        if False:
            return 10
        '\n        Overview:\n            Initialize the Block Model according to input arguments.\n        Arguments:\n            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.\n            - max_T (:obj:`int`): The max context length of the attention, such as 6.\n            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.\n            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.\n        '
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(nn.Linear(h_dim, 4 * h_dim), nn.GELU(), nn.Linear(4 * h_dim, h_dim), nn.Dropout(drop_p))
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Forward computation graph of the decision transformer block, input a sequence tensor             and return a tensor with the same shape.\n        Arguments:\n            - x (:obj:`torch.Tensor`): The input tensor.\n        Returns:\n            - output (:obj:`torch.Tensor`): Output tensor, the shape is the same as the input.\n        Examples:\n            >>> inputs = torch.randn(2, 4, 64)\n            >>> model = Block(64, 5, 4, 0.1)\n            >>> outputs = model(inputs)\n            >>> outputs.shape == torch.Size([2, 4, 64])\n        '
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x

class DecisionTransformer(nn.Module):
    """
    Overview:
        The implementation of decision transformer.
    Interfaces:
        ``__init__``, ``forward``, ``configure_optimizers``
    """

    def __init__(self, state_dim: Union[int, SequenceType], act_dim: int, n_blocks: int, h_dim: int, context_len: int, n_heads: int, drop_p: float, max_timestep: int=4096, state_encoder: Optional[nn.Module]=None, continuous: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Initialize the DecisionTransformer Model according to input arguments.\n        Arguments:\n            - obs_shape (:obj:`Union[int, SequenceType]`): Dimension of state, such as 128 or (4, 84, 84).\n            - act_dim (:obj:`int`): The dimension of actions, such as 6.\n            - n_blocks (:obj:`int`): The number of transformer blocks in the decision transformer, such as 3.\n            - h_dim (:obj:`int`): The dimension of the hidden layers, such as 128.\n            - context_len (:obj:`int`): The max context length of the attention, such as 6.\n            - n_heads (:obj:`int`): The number of heads in calculating attention, such as 8.\n            - drop_p (:obj:`float`): The drop rate of the drop-out layer, such as 0.1.\n            - max_timestep (:obj:`int`): The max length of the total sequence, defaults to be 4096.\n            - state_encoder (:obj:`Optional[nn.Module]`): The encoder to pre-process the given input. If it is set to                 None, the raw state will be pushed into the transformer.\n            - continuous (:obj:`bool`): Whether the action space is continuous, defaults to be ``False``.\n        '
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        input_seq_len = 3 * context_len
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.drop = nn.Dropout(drop_p)
        self.pos_emb = nn.Parameter(torch.zeros(1, input_seq_len + 1, self.h_dim))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, max_timestep + 1, self.h_dim))
        if state_encoder is None:
            self.state_encoder = None
            blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
            self.embed_rtg = torch.nn.Linear(1, h_dim)
            self.embed_state = torch.nn.Linear(state_dim, h_dim)
            self.predict_rtg = torch.nn.Linear(h_dim, 1)
            self.predict_state = torch.nn.Linear(h_dim, state_dim)
            if continuous:
                self.embed_action = torch.nn.Linear(act_dim, h_dim)
                use_action_tanh = True
            else:
                self.embed_action = torch.nn.Embedding(act_dim, h_dim)
                use_action_tanh = False
            self.predict_action = nn.Sequential(*[nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        else:
            blocks = [Block(h_dim, input_seq_len + 1, n_heads, drop_p) for _ in range(n_blocks)]
            self.state_encoder = state_encoder
            self.embed_rtg = nn.Sequential(nn.Linear(1, h_dim), nn.Tanh())
            self.head = nn.Linear(h_dim, act_dim, bias=False)
            self.embed_action = nn.Sequential(nn.Embedding(act_dim, h_dim), nn.Tanh())
        self.transformer = nn.Sequential(*blocks)

    def forward(self, timesteps: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, returns_to_go: torch.Tensor, tar: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Forward computation graph of the decision transformer, input a sequence tensor             and return a tensor with the same shape.\n        Arguments:\n            - timesteps (:obj:`torch.Tensor`): The timestep for input sequence.\n            - states (:obj:`torch.Tensor`): The sequence of states.\n            - actions (:obj:`torch.Tensor`): The sequence of actions.\n            - returns_to_go (:obj:`torch.Tensor`): The sequence of return-to-go.\n            - tar (:obj:`Optional[int]`): Whether to predict action, regardless of index.\n        Returns:\n            - output (:obj:`Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`): Output contains three tensors,             they are correspondingly the predicted states, predicted actions and predicted return-to-go.\n        Examples:\n            >>> B, T = 4, 6\n            >>> state_dim = 3\n            >>> act_dim = 2\n            >>> DT_model = DecisionTransformer(                state_dim=state_dim,                act_dim=act_dim,                n_blocks=3,                h_dim=8,                context_len=T,                n_heads=2,                drop_p=0.1,            )\n            >>> timesteps = torch.randint(0, 100, [B, 3 * T - 1, 1], dtype=torch.long)  # B x T\n            >>> states = torch.randn([B, T, state_dim])  # B x T x state_dim\n            >>> actions = torch.randint(0, act_dim, [B, T, 1])\n            >>> action_target = torch.randint(0, act_dim, [B, T, 1])\n            >>> returns_to_go_sample = torch.tensor([1, 0.8, 0.6, 0.4, 0.2, 0.]).repeat([B, 1]).unsqueeze(-1).float()\n            >>> traj_mask = torch.ones([B, T], dtype=torch.long)  # B x T\n            >>> actions = actions.squeeze(-1)\n            >>> state_preds, action_preds, return_preds = DT_model.forward(                timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go            )\n            >>> assert state_preds.shape == torch.Size([B, T, state_dim])\n            >>> assert return_preds.shape == torch.Size([B, T, 1])\n            >>> assert action_preds.shape == torch.Size([B, T, act_dim])\n        '
        (B, T) = (states.shape[0], states.shape[1])
        if self.state_encoder is None:
            time_embeddings = self.embed_timestep(timesteps)
            state_embeddings = self.embed_state(states) + time_embeddings
            action_embeddings = self.embed_action(actions) + time_embeddings
            returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
            t_p = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
            h = self.embed_ln(t_p)
            h = self.transformer(h)
            h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
            return_preds = self.predict_rtg(h[:, 2])
            state_preds = self.predict_state(h[:, 2])
            action_preds = self.predict_action(h[:, 1])
        else:
            state_embeddings = self.state_encoder(states.reshape(-1, *self.state_dim).type(torch.float32).contiguous())
            state_embeddings = state_embeddings.reshape(B, T, self.h_dim)
            returns_embeddings = self.embed_rtg(returns_to_go.type(torch.float32))
            action_embeddings = self.embed_action(actions.type(torch.long).squeeze(-1))
            token_embeddings = torch.zeros((B, T * 3 - int(tar is None), self.h_dim), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::3, :] = returns_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -T + int(tar is None):, :]
            all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
            position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.h_dim, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
            t_p = token_embeddings + position_embeddings
            h = self.drop(t_p)
            h = self.transformer(h)
            h = self.embed_ln(h)
            logits = self.head(h)
            return_preds = None
            state_preds = None
            action_preds = logits[:, 1::3, :]
        return (state_preds, action_preds, return_preds)

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]=(0.9, 0.95)) -> torch.optim.Optimizer:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            This function returns an optimizer given the input arguments.             We are separating out all parameters of the model into two buckets: those that will experience             weight decay for regularization and those that won't (biases, and layernorm/embedding weights).\n        Arguments:\n            - weight_decay (:obj:`float`): The weigh decay of the optimizer.\n            - learning_rate (:obj:`float`): The learning rate of the optimizer.\n            - betas (:obj:`Tuple[float, float]`): The betas for Adam optimizer.\n        Outputs:\n            - optimizer (:obj:`torch.optim.Optimizer`): The desired optimizer.\n        "
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for (mn, m) in self.named_modules():
            for (pn, p) in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        param_dict = {pn: p for (pn, p) in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, 'parameters %s made it into both decay/no_decay sets!' % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, 'parameters %s were not separated into either decay/no_decay set!' % (str(param_dict.keys() - union_params),)
        optim_groups = [{'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': weight_decay}, {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer