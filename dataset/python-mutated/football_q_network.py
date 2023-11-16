from functools import partial
from ding.utils import deep_merge_dicts, MODEL_REGISTRY
from ding.utils.data import default_collate
from ding.torch_utils import fc_block, Transformer, ResFCBlock, conv2d_block, ResBlock, build_activation, ScatterConnection
import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType, squeeze
from ding.model.common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead
from .football_q_network_default_config import default_model_config

@MODEL_REGISTRY.register('football_naive_q')
class FootballNaiveQ(nn.Module):
    """
        Overview:
            Q model for gfootball.
            utilize the special football obs encoder ``self.football_obs_encoder``: containing
            ``ScalarEncoder``, ``PlayerEncoder`` or ``SpatialEncoder``.
    """

    def __init__(self, cfg: dict={}) -> None:
        if False:
            return 10
        super(FootballNaiveQ, self).__init__()
        self.cfg = deep_merge_dicts(default_model_config, cfg)
        scalar_encoder_arch = self.cfg.encoder.match_scalar
        player_encoder_arch = self.cfg.encoder.player
        self.scalar_encoder = ScalarEncoder(cfg=scalar_encoder_arch)
        self.player_type = player_encoder_arch.encoder_type
        assert self.player_type in ['transformer', 'spatial']
        if self.player_type == 'transformer':
            self.player_encoder = PlayerEncoder(cfg=player_encoder_arch.transformer)
        elif self.player_type == 'spatial':
            self.player_encoder = SpatialEncoder(cfg=player_encoder_arch.spatial)
        scalar_dim = self.scalar_encoder.output_dim
        player_dim = self.player_encoder.output_dim
        head_input_dim = scalar_dim + player_dim
        self.pred_head = FootballHead(input_dim=head_input_dim, cfg=self.cfg.policy)

    def forward(self, x: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Use obs to run MLP or transformer with ``FootballNaiveQ`` and return the prediction dictionary.\n        Arguments:\n            - x (:obj:`Dict`): Dict containing keyword ``processed_obs`` (:obj:`Dict`) and ``raw_obs`` (:obj:`Dict`).\n        Returns:\n            - outputs (:obj:`Dict`): Dict containing keyword ``logit`` (:obj:`torch.Tensor`) and ``action`` (:obj:`torch.Tensor`).\n        Shapes:\n            - x: :math:`(B, N)`, where ``B = batch_size`` and ``N = hidden_size``.\n            - logit: :math:`(B, A)`, where ``A = action_dim``.\n            - action: :math:`(B, )`.\n        '
        if isinstance(x, dict) and len(x) == 2:
            x = x['processed_obs']
        scalar_encodings = self.scalar_encoder(x)
        if self.player_type == 'transformer':
            player_encodings = self.player_encoder(x['players'], x['active_player'])
        elif self.player_type == 'spatial':
            player_encodings = self.player_encoder(x['players'])
        encoding_list = list(scalar_encodings.values()) + [player_encodings]
        x = torch.cat(encoding_list, dim=1)
        x = self.pred_head(x)
        return {'logit': x, 'action': torch.argmax(x, dim=-1)}

class ScalarEncoder(nn.Module):

    def __init__(self, cfg: dict) -> None:
        if False:
            i = 10
            return i + 15
        super(ScalarEncoder, self).__init__()
        self.cfg = cfg
        self.act = nn.ReLU()
        self.output_dim = 0
        for (k, arch) in cfg.items():
            self.output_dim += arch['output_dim']
            encoder = fc_block(arch['input_dim'], arch['output_dim'], activation=self.act)
            setattr(self, k, encoder)

    def forward(self, x: dict) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Shape:\n            - input: dict{scalar_name: scalar_tensor(:math: `(B, scalar_dim)`)}\n            - output: dict{scalar_name: scalar_encoded_tensor(:math: `(B, scalar_encoded_dim)`)}\n        '
        fixed_scalar_sequence = ['ball_position', 'ball_direction', 'ball_rotation', 'ball_owned_team', 'ball_owned_player', 'active_player', 'designated_player', 'active_player_sticky_actions', 'score', 'steps_left', 'game_mode']
        encodings = {}
        for k in fixed_scalar_sequence:
            data = x[k]
            encodings[k] = getattr(self, k)(data)
            if len(encodings[k].shape) == 1:
                encodings[k].unsqueeze_(0)
            elif len(encodings[k].shape) == 3:
                encodings[k].squeeze_(0)
        return encodings

def cat_player_attr(player_data: dict) -> torch.Tensor:
    if False:
        return 10
    '\n    Arguments:\n        player_data: {this_attr_name: [B, this_attr_dim]}\n    Returns:\n        attr: [B, total_attr_dim]\n    '
    fixed_player_attr_sequence = ['team', 'index', 'position', 'direction', 'tired_factor', 'yellow_card', 'active', 'role']
    attr = []
    for k in fixed_player_attr_sequence:
        if len(player_data[k].shape) == 1 and k != 'tired_factor':
            player_data[k].unsqueeze_(0)
        elif len(player_data[k].shape) == 1 and k == 'tired_factor':
            player_data[k].unsqueeze_(-1)
        if len(player_data[k].shape) == 3:
            player_data[k].squeeze_(0)
        attr.append(player_data[k])
    attr = torch.cat(attr, dim=-1)
    return attr

class PlayerEncoder(nn.Module):

    def __init__(self, cfg: dict) -> None:
        if False:
            while True:
                i = 10
        super(PlayerEncoder, self).__init__()
        self.act = nn.ReLU()
        self.player_num = cfg.player_num
        assert self.player_num in [1, 22], self.player_num
        self.output_dim = sum([dim for (k, dim) in cfg.player_attr_dim.items()]) * self.player_num
        player_transformer = Transformer(input_dim=cfg.input_dim, head_dim=cfg.head_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim, head_num=cfg.head_num, mlp_num=cfg.mlp_num, layer_num=cfg.layer_num, dropout_ratio=cfg.dropout_ratio, activation=self.act)
        setattr(self, 'players', player_transformer)

    def forward(self, x: list, active_player: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Shape:\n            - input: list[len=22(=player_num/M)] -> element: dict{attr_name: attr_tensor(:math: `(B, attr_dim)`)}\n            - active_player: :math: `(B, 11)`)\n            - output: :math: `(B, player_num*total_attr_dim)`, player_num is in [1, 22]\n        '
        player_input = self.get_player_input(x, active=active_player)
        player_output = getattr(self, 'players')(player_input)
        player_output = player_output.squeeze(dim=2)
        player_output = player_output.reshape((22, -1, player_output.shape[1]))
        player_output = player_output.permute(1, 0, 2)
        player_output = player_output.reshape((player_output.shape[0], -1))
        return player_output

    def get_player_input(self, data: list, active: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        if self.player_num == 1:
            bs = data[0]['index'].shape[0]
            batch_player = [None for _ in range(bs)]
            for player in data:
                for idx in range(bs):
                    if batch_player[idx] is not None:
                        continue
                    if torch.nonzero(player['index'][idx]).item() == torch.nonzero(active[idx]).item() and torch.nonzero(player['team'][idx]).item() == 0:
                        batch_player[idx] = {k: v[idx] for (k, v) in player.items()}
                if None not in batch_player:
                    break
            batch_player = default_collate(batch_player)
            return cat_player_attr(batch_player).unsqueeze(dim=2)
        elif self.player_num == 22:
            players = []
            for player in data:
                players.append(cat_player_attr(player))
            players = torch.cat(players, dim=0)
            players = players.unsqueeze(dim=2)
            return players

class SpatialEncoder(nn.Module):

    def __init__(self, cfg: dict) -> None:
        if False:
            print('Hello World!')
        super(SpatialEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.norm = cfg.norm_type
        self.scatter = ScatterConnection(cfg.scatter_type)
        input_dim = sum([dim for (k, dim) in cfg.player_attr_dim.items()])
        self.project = conv2d_block(input_dim, cfg.project_dim, 1, 1, 0, activation=self.act, norm_type=self.norm)
        down_layers = []
        dims = [cfg.project_dim] + cfg.down_channels
        self.down_channels = cfg.down_channels
        for i in range(len(self.down_channels)):
            down_layers.append(nn.AvgPool2d(2, 2))
            down_layers.append(conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=self.norm))
        self.downsample = nn.Sequential(*down_layers)
        self.res = nn.ModuleList()
        dim = dims[-1]
        self.resblock_num = cfg.resblock_num
        for i in range(cfg.resblock_num):
            self.res.append(ResBlock(dim, activation=self.act, norm_type=self.norm))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_block(dim, cfg.fc_dim, activation=self.act)
        self.output_dim = cfg.fc_dim

    def forward(self, x: list) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Shape:\n            - input: list[len=22(=player_num/M)] -> element: dict{attr_name: attr_tensor(:math: `(B, attr_dim)`)}\n            - output: :math: `(B, fc_dim)`\n        '
        players = []
        players_loc = []
        granularity = 0.01
        (H, W) = (84, 200)
        for player in x:
            players.append(cat_player_attr(player))
            device = player['position'].device
            player_loc = ((player['position'] + torch.FloatTensor([1.0, 0.42]).to(device)) / granularity).long()
            player_loc_yx = player_loc[:, [1, 0]]
            players_loc.append(player_loc_yx)
        players = torch.stack(players, dim=1)
        players_loc = torch.stack(players_loc, dim=1)
        players_loc[..., 0] = players_loc[..., 0].clamp(0, H - 1)
        players_loc[..., 1] = players_loc[..., 1].clamp(0, W - 1)
        x = self.scatter(players, (H, W), players_loc)
        x = self.project(x)
        x = self.downsample(x)
        for block in self.res:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.shape[:2])
        x = self.fc(x)
        return x

class FootballHead(nn.Module):

    def __init__(self, input_dim: int, cfg: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(FootballHead, self).__init__()
        self.act = nn.ReLU()
        self.input_dim = input_dim
        self.hidden_dim = cfg.res_block.hidden_dim
        self.res_num = cfg.res_block.block_num
        self.dueling = cfg.dqn.dueling
        self.a_layer_num = cfg.dqn.a_layer_num
        self.v_layer_num = cfg.dqn.v_layer_num
        self.action_dim = cfg.action_dim
        self.pre_fc = fc_block(in_channels=input_dim, out_channels=self.hidden_dim, activation=self.act)
        res_blocks_list = []
        for i in range(self.res_num):
            res_blocks_list.append(ResFCBlock(in_channels=self.hidden_dim, activation=self.act, norm_type=None))
        self.res_blocks = nn.Sequential(*res_blocks_list)
        head_fn = partial(DuelingHead, a_layer_num=self.a_layer_num, v_layer_num=self.v_layer_num) if self.dueling else nn.Linear
        self.pred = head_fn(self.hidden_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        "\n        Shape:\n            - input: :math: `(B, input_dim)`), input_dim is the sum of all encoders' output_dim\n            - output: :math: `(B, action_dim)`)\n        "
        x = self.pre_fc(x)
        x = self.res_blocks(x)
        x = self.pred(x)
        return x['logit']