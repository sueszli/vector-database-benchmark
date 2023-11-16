import torch
from typing import Union, Optional
from os import PathLike
from allennlp.fairness.bias_mitigators import HardBiasMitigator, LinearBiasMitigator, INLPBiasMitigator, OSCaRBiasMitigator
from allennlp.fairness.bias_direction_wrappers import BiasDirectionWrapper
from allennlp.fairness.bias_utils import load_word_pairs
from allennlp.common import Registrable
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary

class BiasMitigatorWrapper(Registrable):
    """
    Parent class for bias mitigator wrappers.
    """

    def train(self, mode: bool=True):
        if False:
            return 10
        '\n\n        # Parameters\n\n        mode : `bool`, optional (default=`True`)\n            Sets `requires_grad` to value of `mode` for bias mitigator\n            and associated bias direction.\n        '
        raise NotImplementedError

@BiasMitigatorWrapper.register('hard')
class HardBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction : `BiasDirectionWrapper`
        Bias direction used by mitigator.
    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    equalize_word_pairs_file : `Union[PathLike, str]`
        Path of file containing equalize word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize equalize words.
    mitigator_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of mitigator_vocab to use when tokenizing.
        Disregarded when mitigator_vocab is `None`.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(self, bias_direction: BiasDirectionWrapper, embedding_layer: torch.nn.Embedding, equalize_word_pairs_file: Union[PathLike, str], tokenizer: Tokenizer, mitigator_vocab: Optional[Vocabulary]=None, namespace: str='tokens', requires_grad: bool=True):
        if False:
            print('Hello World!')
        self.bias_direction = bias_direction
        self.predetermined_bias_direction = self.bias_direction(embedding_layer)
        (self.ids1, self.ids2) = load_word_pairs(equalize_word_pairs_file, tokenizer, mitigator_vocab, namespace)
        self.mitigator = HardBiasMitigator(requires_grad=requires_grad)

    def __call__(self, module, module_in, module_out):
        if False:
            i = 10
            return i + 15
        '\n        Called as forward hook.\n        '
        ids1_embeddings = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings = torch.cat(ids1_embeddings)
        ids2_embeddings = torch.cat(ids2_embeddings)
        module_out_size = module_out.size()
        module_out = module_out.flatten(end_dim=-2)
        module_out = self.mitigator(module_out, self.predetermined_bias_direction.to(module_out.device), ids1_embeddings.to(module_out.device), ids2_embeddings.to(module_out.device))[:module_out.size(0)]
        return module_out.reshape(module_out_size)

    def train(self, mode: bool=True):
        if False:
            return 10
        self.mitigator.requires_grad = mode
        self.bias_direction.train(mode)

@BiasMitigatorWrapper.register('linear')
class LinearBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction : `BiasDirectionWrapper`
        Bias direction used by mitigator.
    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(self, bias_direction: BiasDirectionWrapper, embedding_layer: torch.nn.Embedding, requires_grad: bool=True):
        if False:
            while True:
                i = 10
        self.bias_direction = bias_direction
        self.predetermined_bias_direction = self.bias_direction(embedding_layer)
        self.mitigator = LinearBiasMitigator(requires_grad=requires_grad)

    def __call__(self, module, module_in, module_out):
        if False:
            while True:
                i = 10
        '\n        Called as forward hook.\n        '
        module_out_size = module_out.size()
        module_out = module_out.flatten(end_dim=-2)
        module_out = self.mitigator(module_out, self.predetermined_bias_direction.to(module_out.device))
        return module_out.reshape(module_out_size)

    def train(self, mode: bool=True):
        if False:
            print('Hello World!')
        self.mitigator.requires_grad = mode
        self.bias_direction.train(mode)

@BiasMitigatorWrapper.register('inlp')
class INLPBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    mitigator_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of mitigator_vocab to use when tokenizing.
        Disregarded when mitigator_vocab is `None`.
    """

    def __init__(self, embedding_layer: torch.nn.Embedding, seed_word_pairs_file: Union[PathLike, str], tokenizer: Tokenizer, mitigator_vocab: Optional[Vocabulary]=None, namespace: str='tokens'):
        if False:
            i = 10
            return i + 15
        (self.ids1, self.ids2) = load_word_pairs(seed_word_pairs_file, tokenizer, mitigator_vocab, namespace)
        self.mitigator = INLPBiasMitigator()

    def __call__(self, module, module_in, module_out):
        if False:
            i = 10
            return i + 15
        '\n        Called as forward hook.\n        '
        ids1_embeddings = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings = torch.cat(ids1_embeddings)
        ids2_embeddings = torch.cat(ids2_embeddings)
        module_out_size = module_out.size()
        module_out = module_out.flatten(end_dim=-2)
        module_out = self.mitigator(module_out, ids1_embeddings.to(module_out.device), ids2_embeddings.to(module_out.device))
        return module_out.reshape(module_out_size)

    def train(self, mode: bool=True):
        if False:
            return 10
        pass

@BiasMitigatorWrapper.register('oscar')
class OSCaRBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction1 : `BiasDirectionWrapper`
        Bias direction of first concept subspace used by mitigator.
    bias_direction2 : `BiasDirectionWrapper`
        Bias direction of second concept subspace used by mitigator.
    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(self, bias_direction1: BiasDirectionWrapper, bias_direction2: BiasDirectionWrapper, embedding_layer: torch.nn.Embedding, requires_grad: bool=True):
        if False:
            i = 10
            return i + 15
        self.bias_direction1 = bias_direction1
        self.predetermined_bias_direction1 = self.bias_direction1(embedding_layer)
        self.bias_direction2 = bias_direction2(embedding_layer)
        self.predetermined_bias_direction2 = self.bias_direction2(embedding_layer)
        self.mitigator = OSCaRBiasMitigator(requires_grad=requires_grad)

    def __call__(self, module, module_in, module_out):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called as forward hook.\n        '
        module_out_size = module_out.size()
        module_out = module_out.flatten(end_dim=-2)
        module_out = self.mitigator(module_out, self.predetermined_bias_direction1.to(module_out.device), self.predetermined_bias_direction2.to(module_out.device))
        return module_out.reshape(module_out_size)

    def train(self, mode: bool=True):
        if False:
            for i in range(10):
                print('nop')
        self.mitigator.requires_grad = mode
        self.bias_direction1.train(mode)
        self.bias_direction2.train(mode)