import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from TTS.tts.layers.tortoise.arch_utils import AttentionBlock, TypicalLogitsWarper

def null_position_embeddings(range, dim):
    if False:
        return 10
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

def _p(t):
    if False:
        for i in range(10):
            print('nop')
    return t and (len(t), len(t[0]), t[0][0].shape)

class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(chan, chan, kernel_size=3, padding=1), nn.GroupNorm(chan // 8, chan), nn.ReLU(), nn.Conv1d(chan, chan, kernel_size=3, padding=1), nn.GroupNorm(chan // 8, chan))

    def forward(self, x):
        if False:
            while True:
                i = 10
        return F.relu(self.net(x) + x)

class GPT2InferenceModel(GPT2PreTrainedModel):

    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_mel_emb(self, mel_emb):
        if False:
            print('Hello World!')
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if False:
            while True:
                i = 10
        token_type_ids = kwargs.get('token_type_ids', None)
        if not self.kv_cache:
            past_key_values = None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {'input_ids': input_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            for i in range(10):
                print('nop')
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None
        assert labels is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(text_emb.shape[0] // self.cached_mel_emb.shape[0], 0)
            else:
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(attention_mask.shape[1] - mel_len, attention_mask.device)
        transformer_outputs = self.transformer(inputs_embeds=emb, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]
        return CausalLMOutputWithCrossAttentions(loss=None, logits=lm_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions, cross_attentions=transformer_outputs.cross_attentions)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        if False:
            while True:
                i = 10
        '\n        This function is used to re-order the :obj:`past_key_values` cache if\n        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is\n        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.\n        '
        return tuple((tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)) for layer_past in past))

class ConditioningEncoder(nn.Module):

    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, do_checkpointing=False, mean=False):
        if False:
            print('Hello World!')
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        if False:
            return 10
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h[:, :, 0]

class LearnedPositionEmbeddings(nn.Module):

    def __init__(self, seq_len, model_dim, init=0.02):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        if False:
            return 10
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        if False:
            while True:
                i = 10
        return self.emb(torch.arange(0, ind, device=dev))[ind - 1:ind]

def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    if False:
        return 10
    '\n    GPT-2 implemented by the HuggingFace library.\n    '
    from transformers import GPT2Config, GPT2Model
    gpt_config = GPT2Config(vocab_size=256, n_positions=max_mel_seq_len + max_text_seq_len, n_ctx=max_mel_seq_len + max_text_seq_len, n_embd=model_dim, n_layer=layers, n_head=heads, gradient_checkpointing=checkpointing, use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    del gpt.wte
    return (gpt, LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim), None, None)

class MelEncoder(nn.Module):

    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1), nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]), nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1), nn.GroupNorm(channels // 16, channels // 2), nn.ReLU(), nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]), nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1), nn.GroupNorm(channels // 8, channels), nn.ReLU(), nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]))
        self.reduction = 4

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)

class UnifiedVoice(nn.Module):

    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1, mel_length_compression=1024, number_text_tokens=256, start_text_token=None, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193, train_solo_embeddings=False, use_mel_codes_as_input=True, checkpointing=True, types=1):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            layers: Number of layers in transformer stack.\n            model_dim: Operating dimensions of the transformer\n            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64\n            max_text_tokens: Maximum number of text tokens that will be encountered by model.\n            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.\n            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).\n            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.\n            number_text_tokens:\n            start_text_token:\n            stop_text_token:\n            number_mel_codes:\n            start_mel_token:\n            stop_mel_token:\n            train_solo_embeddings:\n            use_mel_codes_as_input:\n            checkpointing:\n        '
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = number_text_tokens * types if start_text_token is None else start_text_token
        self.stop_text_token = 0
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        (self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding) = build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs, self.max_text_tokens + 2, checkpointing)
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def post_init_gpt2_config(self, kv_cache=True):
        if False:
            print('Hello World!')
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(vocab_size=self.max_mel_tokens, n_positions=seq_length, n_ctx=seq_length, n_embd=self.model_dim, n_layer=self.layers, n_head=self.heads, gradient_checkpointing=False, use_cache=True)
        self.inference_model = GPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head, kv_cache=kv_cache)
        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        if False:
            for i in range(10):
                print('nop')
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return (inp, tar)

    def set_mel_padding(self, mel_input_tokens, wav_lengths):
        if False:
            print('Hello World!')
        '\n        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in\n        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required\n        preformatting to create a working TTS model.\n        '
        mel_lengths = torch.div(wav_lengths, self.mel_length_compression, rounding_mode='trunc')
        for b in range(len(mel_lengths)):
            actual_end = mel_lengths[b] + 1
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False):
        if False:
            for i in range(10):
                print('nop')
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions
        enc = gpt_out.last_hidden_state[:, 1:]
        enc = self.final_norm(enc)
        if return_latent:
            return (enc[:, speech_conditioning_inputs.shape[1]:speech_conditioning_inputs.shape[1] + first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:])
        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return (first_logits, second_logits)
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input):
        if False:
            print('Hello World!')
        speech_conditioning_input = speech_conditioning_input.unsqueeze(1) if len(speech_conditioning_input.shape) == 3 else speech_conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1)
        return conds

    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths, types=None, text_first=True, raw_mels=None, return_attentions=False, return_latent=False, clip_inputs=True):
        if False:
            return 10
        '\n        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode\n        (actuated by `text_first`).\n\n        speech_conditioning_input: MEL float tensor, (b,1024)\n        text_inputs: long tensor, (b,t)\n        text_lengths: long tensor, (b,)\n        mel_inputs:  long tensor, (b,m)\n        wav_lengths: long tensor, (b,)\n        raw_mels: MEL float tensor (b,80,s)\n\n        If return_attentions is specified, only logits are returned.\n        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.\n        If clip_inputs is True, the inputs will be clipped to the smallest input size across each input modality.\n        '
        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)
        if clip_inputs:
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            max_mel_len = wav_lengths.max() // self.mel_length_compression
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                raw_mels = raw_mels[:, :, :max_mel_len * 4]
        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)
        conds = speech_conditioning_latent.unsqueeze(1)
        (text_inputs, text_targets) = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        (mel_codes, mel_targets) = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        if raw_mels is not None:
            mel_inp = F.pad(raw_mels, (0, 8))
        else:
            mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)
        if text_first:
            (text_logits, mel_logits) = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return mel_logits[:, :-2]
        else:
            (mel_logits, text_logits) = self.get_logits(conds, mel_emb, self.mel_head, text_emb, self.text_head, get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return text_logits[:, :-2]
        if return_attentions:
            return mel_logits
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return (loss_text.mean(), loss_mel.mean(), mel_logits)

    def inference_speech(self, speech_conditioning_latent, text_inputs, input_tokens=None, num_return_sequences=1, max_generate_length=None, typical_sampling=False, typical_mass=0.9, **hf_generate_kwargs):
        if False:
            print('Hello World!')
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        (text_inputs, text_targets) = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        conds = speech_conditioning_latent.unsqueeze(1)
        emb = torch.cat([conds, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)
        fake_inputs = torch.full((emb.shape[0], conds.shape[1] + emb.shape[1]), fill_value=1, dtype=torch.long, device=text_inputs.device)
        fake_inputs[:, -1] = self.start_mel_token
        trunc_index = fake_inputs.shape[1]
        if input_tokens is None:
            inputs = fake_inputs
        else:
            assert num_return_sequences % input_tokens.shape[0] == 0, 'The number of return sequences must be divisible by the number of input sequences'
            fake_inputs = fake_inputs.repeat(num_return_sequences, 1)
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([fake_inputs, input_tokens], dim=1)
        logits_processor = LogitsProcessorList([TypicalLogitsWarper(mass=typical_mass)]) if typical_sampling else LogitsProcessorList()
        max_length = trunc_index + self.max_mel_tokens - 1 if max_generate_length is None else trunc_index + max_generate_length
        gen = self.inference_model.generate(inputs, bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token, eos_token_id=self.stop_mel_token, max_length=max_length, logits_processor=logits_processor, num_return_sequences=num_return_sequences, **hf_generate_kwargs)
        return gen[:, trunc_index:]
if __name__ == '__main__':
    gpt = UnifiedVoice(model_dim=256, heads=4, train_solo_embeddings=True, use_mel_codes_as_input=True, max_conditioning_inputs=4)
    l = gpt(torch.randn(2, 3, 80, 800), torch.randint(high=120, size=(2, 120)), torch.tensor([32, 120]), torch.randint(high=8192, size=(2, 250)), torch.tensor([250 * 256, 195 * 256]))
    gpt.text_forward(torch.randn(2, 80, 800), torch.randint(high=50, size=(2, 80)), torch.tensor([32, 80]))