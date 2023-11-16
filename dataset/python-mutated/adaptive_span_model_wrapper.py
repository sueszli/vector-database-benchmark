import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.models import FairseqIncrementalDecoder, FairseqLanguageModel, register_model
from .adaptive_span_model import TransformerSeq as AdaptiveSpanTransformerModel
logger = logging.getLogger(__name__)

@dataclass
class AdaptiveSpanSmallConfig(FairseqDataclass):
    vocab_size: int = 50
    d_model: int = 256
    n_head: int = 4
    d_inner: int = 1024
    n_layer: int = 8
    attn_span: int = 1024
    dropout: float = 0.0
    emb_dropout: float = 0.0
    adapt_span_ramp: int = 32
    adapt_span_init: float = 0.0
    aux_loss_scaler: float = 2e-06
    adapt_span_layer: bool = False

@register_model('adaptive_span', dataclass=AdaptiveSpanSmallConfig)
class AdaptiveSpanTransformer(FairseqLanguageModel):

    @classmethod
    def build_model(cls, cfg: AdaptiveSpanSmallConfig, task):
        if False:
            while True:
                i = 10
        return cls(AdaptiveSpanDecoder(cfg, task))

    def get_aux_loss(self):
        if False:
            return 10
        return self.decoder.get_aux_loss()

    def get_current_max_span(self):
        if False:
            i = 10
            return i + 15
        return self.decoder.get_current_max_span()

    def get_current_avg_span(self):
        if False:
            while True:
                i = 10
        return self.decoder.get_current_avg_span()

class AdaptiveSpanDecoder(FairseqIncrementalDecoder):

    def __init__(self, cfg, task):
        if False:
            while True:
                i = 10
        super().__init__(task.target_dictionary)
        self.config = cfg
        config = AdaptiveSpanSmallConfig(vocab_size=len(task.target_dictionary), d_model=cfg.d_model, n_head=cfg.n_head, d_inner=cfg.d_inner, n_layer=cfg.n_layer, attn_span=cfg.attn_span, dropout=cfg.dropout, emb_dropout=cfg.emb_dropout, adapt_span_ramp=cfg.adapt_span_ramp, adapt_span_init=cfg.adapt_span_init, aux_loss_scaler=cfg.aux_loss_scaler, adapt_span_layer=cfg.adapt_span_layer)
        logger.info(config)
        self.model = AdaptiveSpanTransformerModel(**config.__dict__)
        self._mems = None

    def forward(self, src_tokens, incremental_state: Optional[Dict[str, List[torch.Tensor]]]=None, encoder_out=None):
        if False:
            print('Hello World!')
        bsz = src_tokens.size(0)
        if incremental_state is not None:
            mems = self.get_incremental_state('mems')
            src_tokens = src_tokens[:, -1:]
        else:
            mems = self._mems
        if mems is None:
            mems = self.init_hid_cache(bsz)
        output = self.model(x=src_tokens, h_cache=mems)
        if incremental_state is not None:
            self.set_incremental_state(incremental_state, 'mems', output[1])
        else:
            self._mems = output[1]
        return (output[0],)

    def max_positions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.attn_span

    def init_hid_cache(self, batch_sz):
        if False:
            print('Hello World!')
        hid = []
        for layer in self.model.layers:
            param = next(self.model.parameters())
            h = torch.zeros(batch_sz, layer.get_cache_size(), self.config.d_model, dtype=param.dtype, device=param.device)
            hid.append(h)
        return hid

    def get_aux_loss(self):
        if False:
            while True:
                i = 10
        return self.model.get_aux_loss()

    def get_current_max_span(self):
        if False:
            for i in range(10):
                print('nop')
        return self.model.get_current_max_span()

    def get_current_avg_span(self):
        if False:
            i = 10
            return i + 15
        return self.model.get_current_avg_span()

    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]], new_order: torch.Tensor):
        if False:
            print('Hello World!')
        'Reorder incremental state.\n\n        This will be called when the order of the input has changed from the\n        previous time step. A typical use case is beam search, where the input\n        order changes between time steps based on the selection of beams.\n        '
        raise NotImplementedError('This is required for generation/beam search')