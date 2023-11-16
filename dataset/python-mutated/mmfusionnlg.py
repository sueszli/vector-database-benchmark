import torch
from torch.nn import functional as F
from typing import Optional, Iterable
try:
    from transformers import BertPreTrainedModel
    from transformers.modeling_bert import BertOnlyMLMHead
    from transformers.file_utils import ModelOutput
    from transformers.modeling_outputs import CausalLMOutput
    from transformers.generation_utils import BeamHypotheses, top_k_top_p_filtering
except ImportError:
    pass
from .mmfusion import MMFusion
from .transformermodel import MMBertModel
from ..modules import VideoTokenMLP

class MMFusionNLG(MMFusion):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        if config.model.max_decode_length is not None:
            self.max_length = min(config.model.max_decode_length, config.dataset.max_len - config.dataset.max_video_len - 3)
        else:
            self.max_length = config.dataset.max_len - config.dataset.max_video_len - 3
        self.gen_param = config.gen_param if config.gen_param is not None else {}

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask, video_label=None, text_label=None, **kwargs):
        if False:
            return 10
        'use pre-trained LM header for generation.'
        (attention_mask, token_type_ids) = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, masked_lm_labels=text_label)
        return {'logits': outputs[0]}

    @torch.no_grad()
    def generate(self, caps, cmasks, vfeats, vmasks, attention_mask=None, bos_token_id=None, eos_token_id=None, **kwargs):
        if False:
            return 10
        assert caps.size(1) == 3
        (attention_mask, token_type_ids) = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        output = self.mm_encoder.generate(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, bos_token_id=bos_token_id, eos_token_id=eos_token_id, max_length=self.max_length, **self.gen_param)
        return output

class MMBertForNLG(BertPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.bert = MMBertModel(config)
        self.videomlp = VideoTokenMLP(config)
        self.cls = BertOnlyMLMHead(config)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, input_video_embeds=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, masked_lm_labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            print('Hello World!')
        video_tokens = self.videomlp(input_video_embeds)
        outputs = self.bert(input_ids, video_tokens, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = None
        if masked_lm_labels is not None:
            text_offset = input_video_embeds.size(1) + 1
            text_sequence_output = torch.cat([sequence_output[:, :1], sequence_output[:, text_offset:]], dim=1)
            hidden_size = text_sequence_output.size(-1)
            labels_mask = masked_lm_labels != -100
            selected_text_output = text_sequence_output.masked_select(labels_mask.unsqueeze(-1)).view(-1, hidden_size)
            prediction_scores = self.cls(selected_text_output)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return output
        text_offset = input_video_embeds.size(1) + 2
        text_sequence_output = sequence_output[:, text_offset:]
        prediction_scores = self.cls(text_sequence_output)
        return CausalLMOutput(loss=None, logits=prediction_scores)

    def prepare_inputs_for_generation(self, input_ids, input_video_embeds, attention_mask=None, token_type_ids=None, **model_kwargs):
        if False:
            i = 10
            return i + 15
        seq_len = input_ids.size(1) + input_video_embeds.size(1)
        if attention_mask is not None:
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, :seq_len, :seq_len]
            elif len(attention_mask.size()) == 3:
                attention_mask = attention_mask[:, :seq_len, :seq_len]
            else:
                attention_mask = attention_mask[:, :seq_len]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :seq_len]
        return {'input_ids': input_ids, 'input_video_embeds': input_video_embeds, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, max_length: Optional[int]=None, min_length: Optional[int]=None, do_sample: Optional[bool]=None, early_stopping: Optional[bool]=None, num_beams: Optional[int]=None, temperature: Optional[float]=None, top_k: Optional[int]=None, top_p: Optional[float]=None, repetition_penalty: Optional[float]=None, bad_words_ids: Optional[Iterable[int]]=None, bos_token_id: Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[int]=None, length_penalty: Optional[float]=None, no_repeat_ngram_size: Optional[int]=None, num_return_sequences: Optional[int]=None, attention_mask: Optional[torch.LongTensor]=None, decoder_start_token_id: Optional[int]=None, use_cache: Optional[bool]=None, **model_kwargs) -> torch.LongTensor:
        if False:
            i = 10
            return i + 15
        '\n        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,\n        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.\n        Adapted in part from `Facebook\'s XLM beam search code\n        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.\n        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the\n        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values\n        indicated are the default values of those config.\n        Most of these parameters are explained in more detail in `this blog post\n        <https://huggingface.co/blog/how-to-generate>`__.\n        Parameters:\n            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n                The sequence used as a prompt for the generation. If :obj:`None` the method initializes\n                it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.\n            decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n                initial input_ids for the decoder of encoder-decoder type models. If :obj:`None` then only\n                decoder_start_token_id is passed as the first token to the decoder.\n            max_length (:obj:`int`, `optional`, defaults to 20):\n                The maximum length of the sequence to be generated.\n            min_length (:obj:`int`, `optional`, defaults to 10):\n                The minimum length of the sequence to be generated.\n            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):\n                Whether or not to use sampling ; use greedy decoding otherwise.\n            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):\n                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.\n            num_beams (:obj:`int`, `optional`, defaults to 1):\n                Number of beams for beam search. 1 means no beam search.\n            temperature (:obj:`float`, `optional`, defaults tp 1.0):\n                The value used to module the next token probabilities.\n            top_k (:obj:`int`, `optional`, defaults to 50):\n                The number of highest probability vocabulary tokens to keep for top-k-filtering.\n            top_p (:obj:`float`, `optional`, defaults to 1.0):\n                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or\n                higher are kept for generation.\n            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):\n                The parameter for repetition penalty. 1.0 means no penalty. See `this paper\n                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.\n            pad_token_id (:obj:`int`, `optional`):\n                The id of the `padding` token.\n            bos_token_id (:obj:`int`, `optional`):\n                The id of the `beginning-of-sequence` token.\n            eos_token_id (:obj:`int`, `optional`):\n                The id of the `end-of-sequence` token.\n            length_penalty (:obj:`float`, `optional`, defaults to 1.0):\n                Exponential penalty to the length. 1.0 means no penalty.\n                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in\n                order to encourage the model to produce longer sequences.\n            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):\n                If set to int > 0, all ngrams of that size can only occur once.\n            bad_words_ids(:obj:`List[int]`, `optional`):\n                List of token ids that are not allowed to be generated. In order to get the tokens of the words that\n                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.\n            num_return_sequences(:obj:`int`, `optional`, defaults to 1):\n                The number of independently computed returned sequences for each element in the batch.\n            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for\n                tokens that are not masked, and 0 for masked tokens.\n                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.\n                `What are attention masks? <../glossary.html#attention-mask>`__\n            decoder_start_token_id (:obj:`int`, `optional`):\n                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.\n            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):\n                Whether or not the model should use the past last key/values attentions (if applicable to the model) to\n                speed up decoding.\n            model_kwargs:\n                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.\n        Return:\n            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:\n            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or\n            shorter if all batches finished early due to the :obj:`eos_token_id`.\n        Examples::\n            tokenizer = AutoTokenizer.from_pretrained(\'distilgpt2\')   # Initialize tokenizer\n            model = AutoModelWithLMHead.from_pretrained(\'distilgpt2\')    # Download model and configuration from S3 and cache.\n            outputs = model.generate(max_length=40)  # do greedy decoding\n            print(\'Generated: {}\'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))\n            tokenizer = AutoTokenizer.from_pretrained(\'openai-gpt\')   # Initialize tokenizer\n            model = AutoModelWithLMHead.from_pretrained(\'openai-gpt\')    # Download model and configuration from S3 and cache.\n            input_context = \'The dog\'\n            input_ids = tokenizer.encode(input_context, return_tensors=\'pt\')  # encode input context\n            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context \'The dog\'\n            for i in range(3): #  3 output sequences were generated\n                print(\'Generated {}: {}\'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))\n            tokenizer = AutoTokenizer.from_pretrained(\'distilgpt2\')   # Initialize tokenizer\n            model = AutoModelWithLMHead.from_pretrained(\'distilgpt2\')    # Download model and configuration from S3 and cache.\n            input_context = \'The dog\'\n            input_ids = tokenizer.encode(input_context, return_tensors=\'pt\')  # encode input context\n            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling\n            for i in range(3): #  3 output sequences were generated\n                print(\'Generated {}: {}\'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))\n            tokenizer = AutoTokenizer.from_pretrained(\'ctrl\')   # Initialize tokenizer\n            model = AutoModelWithLMHead.from_pretrained(\'ctrl\')    # Download model and configuration from S3 and cache.\n            input_context = \'Legal My neighbor is\'  # "Legal" is one of the control codes for ctrl\n            input_ids = tokenizer.encode(input_context, return_tensors=\'pt\')  # encode input context\n            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences\n            print(\'Generated: {}\'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))\n            tokenizer = AutoTokenizer.from_pretrained(\'gpt2\')   # Initialize tokenizer\n            model = AutoModelWithLMHead.from_pretrained(\'gpt2\')    # Download model and configuration from S3 and cache.\n            input_context = \'My cute dog\'  # "Legal" is one of the control codes for ctrl\n            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in [\'idiot\', \'stupid\', \'shut up\']]\n            input_ids = tokenizer.encode(input_context, return_tensors=\'pt\')  # encode input context\n            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated\n        '
        if self.get_output_embeddings() is None:
            raise AttributeError('You tried to generate sequences with a model that does not have a LM Head.Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )')
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1
        assert isinstance(max_length, int) and max_length > 0, '`max_length` should be a strictly positive integer.'
        assert isinstance(min_length, int) and min_length >= 0, '`min_length` should be a positive integer.'
        assert isinstance(do_sample, bool), '`do_sample` should be a boolean.'
        assert isinstance(early_stopping, bool), '`early_stopping` should be a boolean.'
        assert isinstance(use_cache, bool), '`use_cache` should be a boolean.'
        assert isinstance(num_beams, int) and num_beams > 0, '`num_beams` should be a strictly positive integer.'
        assert temperature > 0, '`temperature` should be strictly positive.'
        assert isinstance(top_k, int) and top_k >= 0, '`top_k` should be a positive integer.'
        assert 0 <= top_p <= 1, '`top_p` should be between 0 and 1.'
        assert repetition_penalty >= 1.0, '`repetition_penalty` should be >= 1.'
        assert input_ids is not None or (isinstance(bos_token_id, int) and bos_token_id >= 0), 'If input_ids is not defined, `bos_token_id` should be a positive integer.'
        assert pad_token_id is None or (isinstance(pad_token_id, int) and pad_token_id >= 0), '`pad_token_id` should be a positive integer.'
        assert eos_token_id is None or (isinstance(eos_token_id, int) and eos_token_id >= 0), '`eos_token_id` should be a positive integer.'
        assert length_penalty > 0, '`length_penalty` should be strictly positive.'
        assert isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0, '`no_repeat_ngram_size` should be a positive integer.'
        assert isinstance(num_return_sequences, int) and num_return_sequences > 0, '`num_return_sequences` should be a strictly positive integer.'
        assert bad_words_ids is None or (isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)), '`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated'
        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, 'you should either supply a context to complete as `input_ids` input or a `bos_token_id` (integer >= 0) as a first token to start the generation.'
            input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device)
        else:
            assert input_ids.dim() == 2, 'Input prompt should be of shape (batch_size, sequence length).'
        if do_sample is False:
            if num_beams == 1:
                assert num_return_sequences == 1, 'Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1'
            else:
                assert num_beams >= num_return_sequences, 'Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences'
        if attention_mask is None and pad_token_id is not None and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        if pad_token_id is None and eos_token_id is not None:
            print('Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence'.format(eos_token_id))
            pad_token_id = eos_token_id
        if hasattr(self.config, 'vocab_size'):
            vocab_size = self.config.vocab_size
        elif self.config.is_encoder_decoder and hasattr(self.config, 'decoder') and hasattr(self.config.decoder, 'vocab_size'):
            vocab_size = self.config.decoder.vocab_size
        else:
            raise ValueError('either self.config.vocab_size or self.config.decoder.vocab_size needs to be defined')
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1
        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                if bos_token_id is not None:
                    decoder_start_token_id = bos_token_id
                elif hasattr(self.config, 'decoder') and hasattr(self.config.decoder, 'bos_token_id') and (self.config.decoder.bos_token_id is not None):
                    decoder_start_token_id = self.config.decoder.bos_token_id
                else:
                    raise ValueError('decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation')
            assert hasattr(self, 'get_encoder'), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), '{} should be a method'.format(self.get_encoder)
            encoder = self.get_encoder()
            encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
        if num_return_sequences > 1 or num_beams > 1:
            input_video_embeds = model_kwargs.pop('input_video_embeds', None)
            token_type_ids = model_kwargs.pop('token_type_ids', None)
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            (input_video_embeds_len, input_video_embeds_hidden) = (input_video_embeds.size(1), input_video_embeds.size(2))
            input_video_embeds = input_video_embeds.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_video_embeds_len, input_video_embeds_hidden)
            (attention_mask_from_len, attention_mask_to_len) = (attention_mask.size(1), attention_mask.size(2))
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, attention_mask_from_len, attention_mask_to_len)
            token_type_ids_len = token_type_ids.size(1)
            token_type_ids = token_type_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, token_type_ids_len)
            input_ids = input_ids.contiguous().view(effective_batch_size * num_beams, input_ids_len)
            input_video_embeds = input_video_embeds.contiguous().view(effective_batch_size * num_beams, input_video_embeds_len, input_video_embeds_hidden)
            attention_mask = attention_mask.contiguous().view(effective_batch_size * num_beams, attention_mask_from_len, attention_mask_to_len)
            token_type_ids = token_type_ids.contiguous().view(effective_batch_size * num_beams, token_type_ids_len)
            model_kwargs['input_video_embeds'] = input_video_embeds
            model_kwargs['token_type_ids'] = token_type_ids
        if self.config.is_encoder_decoder:
            device = next(self.parameters()).device
            if decoder_input_ids is not None:
                input_ids = decoder_input_ids.repeat(effective_batch_size * num_beams, 1).to(device)
            else:
                input_ids = torch.full((effective_batch_size * num_beams, 1), decoder_start_token_id, dtype=torch.long, device=device)
            cur_len = input_ids.shape[-1]
            assert batch_size == encoder_outputs.last_hidden_state.shape[0], f'expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} '
            expanded_batch_idxs = torch.arange(batch_size).view(-1, 1).repeat(1, num_beams * effective_batch_mult).view(-1).to(input_ids.device)
            encoder_outputs['last_hidden_state'] = encoder_outputs.last_hidden_state.index_select(0, expanded_batch_idxs)
            model_kwargs['encoder_outputs'] = encoder_outputs
        else:
            cur_len = input_ids.shape[-1]
        assert cur_len < max_length, f'The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`'
        if num_beams > 1:
            output = self._generate_beam_search(input_ids, cur_len=cur_len, max_length=max_length, min_length=min_length, do_sample=do_sample, early_stopping=early_stopping, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id, batch_size=effective_batch_size, num_return_sequences=num_return_sequences, length_penalty=length_penalty, num_beams=num_beams, vocab_size=vocab_size, attention_mask=attention_mask, use_cache=use_cache, model_kwargs=model_kwargs)
        else:
            output = self._generate_no_beam_search(input_ids, cur_len=cur_len, max_length=max_length, min_length=min_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id, batch_size=effective_batch_size, attention_mask=attention_mask, use_cache=use_cache, model_kwargs=model_kwargs)
        return output

    def _generate_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, early_stopping, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id, batch_size, num_return_sequences, length_penalty, num_beams, vocab_size, attention_mask, use_cache, model_kwargs):
        if False:
            print('Hello World!')
        'Generate sequences for each example with beam search.'
        generated_hyps = [BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping) for _ in range(batch_size)]
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        if do_sample is False:
            beam_scores[:, 1:] = -1000000000.0
        beam_scores = beam_scores.view(-1)
        past = None
        done = [False for _ in range(batch_size)]
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            if 'past_key_values' in outputs:
                past = outputs.past_key_values
            elif 'mems' in outputs:
                past = outputs.mems
            if self.config.is_encoder_decoder and do_sample is False:
                next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len, max_length=max_length)
            scores = F.log_softmax(next_token_logits, dim=-1)
            scores = self.postprocess_next_token_scores(scores=scores, input_ids=input_ids, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, cur_len=cur_len, min_length=min_length, max_length=max_length, eos_token_id=eos_token_id, repetition_penalty=repetition_penalty, batch_size=batch_size, num_beams=num_beams)
            assert scores.shape == (batch_size * num_beams, vocab_size), 'Shapes of scores: {} != {}'.format(scores.shape, (batch_size * num_beams, vocab_size))
            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)
                if temperature != 1.0:
                    _scores = _scores / temperature
                _scores = top_k_top_p_filtering(_scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2)
                _scores = _scores.contiguous().view(batch_size, num_beams * vocab_size)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
                next_scores = torch.gather(_scores, -1, next_tokens)
                (next_scores, next_scores_indices) = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)
            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)
                next_scores = next_scores.view(batch_size, num_beams * vocab_size)
                (next_scores, next_tokens) = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
            next_batch_beam = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    assert len(generated_hyps[batch_idx]) >= num_beams, 'Batch can only be done if at least {} beams have been generated'.format(num_beams)
                    assert eos_token_id is not None and pad_token_id is not None, 'generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)
                    continue
                next_sent_beam = []
                for (beam_token_rank, (beam_token_id, beam_token_score)) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx])):
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size
                    effective_beam_id = batch_idx * num_beams + beam_id
                    if eos_token_id is not None and token_id.item() == eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(input_ids[effective_beam_id].clone(), beam_token_score.item())
                    else:
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    if len(next_sent_beam) == num_beams:
                        break
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(next_scores[batch_idx].max().item(), cur_len)
                assert len(next_sent_beam) == num_beams, 'Beam should always be full'
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), 'We should have added num_beams each step'
            if all(done):
                break
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
            if past is not None:
                past = self._reorder_cache(past, beam_idx)
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            if eos_token_id is not None and all(((token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx])):
                assert torch.all(next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]), 'If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}'.format(next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx])
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        for (i, hypotheses) in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len)
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, '`pad_token_id` has to be defined'
            decoded.fill_(pad_token_id)
        for (i, hypo) in enumerate(best):
            decoded[i, :sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return decoded

    def _generate_no_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id, batch_size, attention_mask, use_cache, model_kwargs):
        if False:
            while True:
                i = 10
        'Generate sequences for each example without beam search (num_beams == 1).\n        All returned sequence are generated independantly.\n        '
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        past = None
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            scores = self.postprocess_next_token_scores(scores=next_token_logits, input_ids=input_ids, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, cur_len=cur_len, min_length=min_length, max_length=max_length, eos_token_id=eos_token_id, repetition_penalty=repetition_penalty, batch_size=batch_size, num_beams=1)
            if 'past_key_values' in outputs:
                past = outputs.past_key_values
            elif 'mems' in outputs:
                past = outputs.mems
            if do_sample:
                if temperature != 1.0:
                    scores = scores / temperature
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            if eos_token_id is not None:
                tokens_to_add = next_token * unfinished_sents + pad_token_id * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1
            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                unfinished_sents.mul_((~eos_in_sents).long())
            if unfinished_sents.max() == 0:
                break
        return input_ids