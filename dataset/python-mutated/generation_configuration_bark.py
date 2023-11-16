""" BARK model generation configuration"""
import copy
from typing import Dict
from ...generation.configuration_utils import GenerationConfig
from ...utils import logging
logger = logging.get_logger(__name__)

class BarkSemanticGenerationConfig(GenerationConfig):
    model_type = 'semantic'

    def __init__(self, eos_token_id=10000, renormalize_logits=True, max_new_tokens=768, output_scores=False, return_dict_in_generate=False, output_hidden_states=False, output_attentions=False, temperature=1.0, do_sample=False, text_encoding_offset=10048, text_pad_token=129595, semantic_infer_token=129599, semantic_vocab_size=10000, max_input_semantic_length=256, semantic_rate_hz=49.9, min_eos_p=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Class that holds a generation configuration for [`BarkSemanticModel`].\n\n        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the\n        documentation from [`GenerationConfig`] for more information.\n\n        Args:\n            eos_token_id (`int`, *optional*, defaults to 10_000):\n                The id of the *end-of-sequence* token.\n            renormalize_logits (`bool`, *optional*, defaults to `True`):\n                Whether to renormalize the logits after applying all the logits processors or warpers (including the\n                custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the\n                score logits are normalized but some logit processors or warpers break the normalization.\n            max_new_tokens (`int`, *optional*, defaults to 768):\n                The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.\n            output_scores (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.\n            return_dict_in_generate (`bool`, *optional*, defaults to `False`):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n            output_hidden_states (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more details.\n            output_attentions (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more details.\n            temperature (`float`, *optional*, defaults to 1.0):\n                The value used to modulate the next token probabilities.\n            do_sample (`bool`, *optional*, defaults to `False`):\n                Whether or not to use sampling ; use greedy decoding otherwise.\n            text_encoding_offset (`int`, *optional*, defaults to 10_048):\n                Text encoding offset.\n            text_pad_token (`int`, *optional*, defaults to 129_595):\n                Text pad token.\n            semantic_infer_token (`int`, *optional*, defaults to 129_599):\n                Semantic infer token.\n            semantic_vocab_size (`int`, *optional*, defaults to 10_000):\n                Semantic vocab size.\n            max_input_semantic_length (`int`, *optional*, defaults to 256):\n                Max length of semantic input vector.\n            semantic_rate_hz (`float`, *optional*, defaults to 49.9):\n                Semantic rate in Hertz.\n            min_eos_p (`float`, *optional*):\n                Minimum threshold of the probability of the EOS token for it to be sampled. This is an early stopping\n                strategy to mitigate potential unwanted generations at the end of a prompt. The original implementation\n                suggests a default value of 0.2.\n        "
        super().__init__(temperature=temperature, do_sample=do_sample, eos_token_id=eos_token_id, renormalize_logits=renormalize_logits, max_new_tokens=max_new_tokens, output_scores=output_scores, return_dict_in_generate=return_dict_in_generate, output_hidden_states=output_hidden_states, output_attentions=output_attentions, **kwargs)
        self.text_encoding_offset = text_encoding_offset
        self.text_pad_token = text_pad_token
        self.semantic_pad_token = eos_token_id
        self.semantic_infer_token = semantic_infer_token
        self.semantic_vocab_size = semantic_vocab_size
        self.max_input_semantic_length = max_input_semantic_length
        self.semantic_rate_hz = semantic_rate_hz
        self.min_eos_p = min_eos_p

class BarkCoarseGenerationConfig(GenerationConfig):
    model_type = 'coarse_acoustics'

    def __init__(self, renormalize_logits=True, output_scores=False, return_dict_in_generate=False, output_hidden_states=False, output_attentions=False, temperature=1.0, do_sample=False, coarse_semantic_pad_token=12048, coarse_rate_hz=75, n_coarse_codebooks=2, coarse_infer_token=12050, max_coarse_input_length=256, max_coarse_history: int=630, sliding_window_len: int=60, **kwargs):
        if False:
            while True:
                i = 10
        "Class that holds a generation configuration for [`BarkCoarseModel`].\n\n        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the\n        documentation from [`GenerationConfig`] for more information.\n\n        Args:\n            renormalize_logits (`bool`, *optional*, defaults to `True`):\n                Whether to renormalize the logits after applying all the logits processors or warpers (including the\n                custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the\n                score logits are normalized but some logit processors or warpers break the normalization.\n            output_scores (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.\n            return_dict_in_generate (`bool`, *optional*, defaults to `False`):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n            output_hidden_states (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more details.\n            output_attentions (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more details.\n            temperature (`float`, *optional*, defaults to 1.0):\n                The value used to modulate the next token probabilities.\n            do_sample (`bool`, *optional*, defaults to `False`):\n                Whether or not to use sampling ; use greedy decoding otherwise.\n            coarse_semantic_pad_token (`int`, *optional*, defaults to 12_048):\n                Coarse semantic pad token.\n            coarse_rate_hz (`int`, *optional*, defaults to 75):\n                Coarse rate in Hertz.\n            n_coarse_codebooks (`int`, *optional*, defaults to 2):\n                Number of coarse codebooks.\n            coarse_infer_token (`int`, *optional*, defaults to 12_050):\n                Coarse infer token.\n            max_coarse_input_length (`int`, *optional*, defaults to 256):\n                Max length of input coarse vector.\n            max_coarse_history (`int`, *optional*, defaults to 630):\n                Max length of the output of the coarse acoustics model used in the fine generation step.\n            sliding_window_len (`int`, *optional*, defaults to 60):\n                The coarse generation step uses a sliding window to generate raw audio.\n        "
        super().__init__(temperature=temperature, do_sample=do_sample, renormalize_logits=renormalize_logits, output_scores=output_scores, return_dict_in_generate=return_dict_in_generate, output_hidden_states=output_hidden_states, output_attentions=output_attentions, **kwargs)
        self.coarse_semantic_pad_token = coarse_semantic_pad_token
        self.coarse_rate_hz = coarse_rate_hz
        self.n_coarse_codebooks = n_coarse_codebooks
        self.coarse_infer_token = coarse_infer_token
        self.max_coarse_input_length = max_coarse_input_length
        self.max_coarse_history = max_coarse_history
        self.sliding_window_len = sliding_window_len

class BarkFineGenerationConfig(GenerationConfig):
    model_type = 'fine_acoustics'

    def __init__(self, temperature=1.0, max_fine_history_length=512, max_fine_input_length=1024, n_fine_codebooks=8, **kwargs):
        if False:
            print('Hello World!')
        'Class that holds a generation configuration for [`BarkFineModel`].\n\n        [`BarkFineModel`] is an autoencoder model, so should not usually be used for generation. However, under the\n        hood, it uses `temperature` when used by [`BarkModel`]\n\n        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the\n        documentation from [`GenerationConfig`] for more information.\n\n        Args:\n            temperature (`float`, *optional*):\n                The value used to modulate the next token probabilities.\n            max_fine_history_length (`int`, *optional*, defaults to 512):\n                Max length of the fine history vector.\n            max_fine_input_length (`int`, *optional*, defaults to 1024):\n                Max length of fine input vector.\n            n_fine_codebooks (`int`, *optional*, defaults to 8):\n                Number of codebooks used.\n        '
        super().__init__(temperature=temperature)
        self.max_fine_history_length = max_fine_history_length
        self.max_fine_input_length = max_fine_input_length
        self.n_fine_codebooks = n_fine_codebooks

    def validate(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Overrides GenerationConfig.validate because BarkFineGenerationConfig don't use any parameters outside\n        temperature.\n        "
        pass

class BarkGenerationConfig(GenerationConfig):
    model_type = 'bark'
    is_composition = True

    def __init__(self, semantic_config: Dict=None, coarse_acoustics_config: Dict=None, fine_acoustics_config: Dict=None, sample_rate=24000, codebook_size=1024, **kwargs):
        if False:
            return 10
        'Class that holds a generation configuration for [`BarkModel`].\n\n        The [`BarkModel`] does not have a `generate` method, but uses this class to generate speeches with a nested\n        [`BarkGenerationConfig`] which uses [`BarkSemanticGenerationConfig`], [`BarkCoarseGenerationConfig`],\n        [`BarkFineGenerationConfig`].\n\n        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the\n        documentation from [`GenerationConfig`] for more information.\n\n        Args:\n            semantic_config (`Dict`, *optional*):\n                Semantic generation configuration.\n            coarse_acoustics_config (`Dict`, *optional*):\n                Coarse generation configuration.\n            fine_acoustics_config (`Dict`, *optional*):\n                Fine generation configuration.\n            sample_rate (`int`, *optional*, defaults to 24_000):\n                Sample rate.\n            codebook_size (`int`, *optional*, defaults to 1024):\n                Vector length for each codebook.\n        '
        if semantic_config is None:
            semantic_config = {}
            logger.info('semantic_config is None. initializing the semantic model with default values.')
        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info('coarse_acoustics_config is None. initializing the coarse model with default values.')
        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info('fine_acoustics_config is None. initializing the fine model with default values.')
        self.semantic_config = BarkSemanticGenerationConfig(**semantic_config)
        self.coarse_acoustics_config = BarkCoarseGenerationConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkFineGenerationConfig(**fine_acoustics_config)
        self.sample_rate = sample_rate
        self.codebook_size = codebook_size

    @classmethod
    def from_sub_model_configs(cls, semantic_config: BarkSemanticGenerationConfig, coarse_acoustics_config: BarkCoarseGenerationConfig, fine_acoustics_config: BarkFineGenerationConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Instantiate a [`BarkGenerationConfig`] (or a derived class) from bark sub-models generation configuration.\n\n        Returns:\n            [`BarkGenerationConfig`]: An instance of a configuration object\n        '
        return cls(semantic_config=semantic_config.to_dict(), coarse_acoustics_config=coarse_acoustics_config.to_dict(), fine_acoustics_config=fine_acoustics_config.to_dict(), **kwargs)

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].\n\n        Returns:\n            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,\n        '
        output = copy.deepcopy(self.__dict__)
        output['semantic_config'] = self.semantic_config.to_dict()
        output['coarse_acoustics_config'] = self.coarse_acoustics_config.to_dict()
        output['fine_acoustics_config'] = self.fine_acoustics_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output