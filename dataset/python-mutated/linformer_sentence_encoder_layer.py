import torch
from fairseq import utils
from fairseq.modules import TransformerEncoderLayer
from .multihead_linear_attention import MultiheadLinearAttention

class LinformerTransformerEncoderLayer(TransformerEncoderLayer):
    """
    Implements a Linformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, args, shared_compress_layer):
        if False:
            for i in range(10):
                print('nop')
        self.shared_compress_layer = [shared_compress_layer]
        super().__init__(args)
        self.register_buffer('version', torch.tensor(2))

    def build_self_attention(self, embed_dim, args):
        if False:
            return 10
        return MultiheadLinearAttention(embed_dim, args.encoder_attention_heads, dropout=args.dropout, self_attention=True, q_noise=args.quant_noise_pq, qn_block_size=args.quant_noise_pq_block_size, compressed=args.compressed, max_seq_len=args.max_positions, shared_kv_compressed=args.shared_kv_compressed, shared_compress_layer=self.shared_compress_layer[0], freeze_compress=args.freeze_compress)

    def upgrade_state_dict_named(self, state_dict, name):
        if False:
            for i in range(10):
                print('nop')
        super().upgrade_state_dict_named(state_dict, name)
        prefix = name + '.' if name != '' else ''
        if utils.item(state_dict.get(f'{prefix}version', torch.tensor(1))) < 2:
            state_dict[f'{prefix}version'] = torch.tensor(1)
            if f'{prefix}shared_compress_layer.weight' in state_dict:
                self.shared_compress_layer = [torch.nn.Linear(self.shared_compress_layer[0].weight.size(1), self.shared_compress_layer[0].weight.size(0))]
                self.self_attn = self.build_self_attention(self.embed_dim, self.args)
                del state_dict[f'{prefix}shared_compress_layer.weight']
                if f'{prefix}shared_compress_layer.bias' in state_dict:
                    del state_dict[f'{prefix}shared_compress_layer.bias']