import inspect
from typing import Any, Dict, List
from fairseq import utils
from fairseq.logging import metrics
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from torch.nn.modules.loss import _Loss

class FairseqCriterion(_Loss):

    def __init__(self, task):
        if False:
            return 10
        super().__init__()
        self.task = task
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100

    @classmethod
    def add_args(cls, parser):
        if False:
            i = 10
            return i + 15
        'Add criterion-specific arguments to the parser.'
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_criterion(cls, cfg: FairseqDataclass, task):
        if False:
            for i in range(10):
                print('nop')
        'Construct a criterion from command-line args.'
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if p.kind == p.POSITIONAL_ONLY or p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD:
                raise NotImplementedError('{} not supported'.format(p.kind))
            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
            if p.name == 'task':
                init_args['task'] = task
            elif p.name == 'cfg':
                init_args['cfg'] = cfg
            elif hasattr(cfg, p.name):
                init_args[p.name] = getattr(cfg, p.name)
            elif p.default != p.empty:
                pass
            else:
                raise NotImplementedError('Unable to infer Criterion arguments, please implement {}.build_criterion'.format(cls.__name__))
        return cls(**init_args)

    def forward(self, model, sample, reduce=True):
        if False:
            print('Hello World!')
        'Compute the loss for the given sample.\n\n        Returns a tuple with three elements:\n        1) the loss\n        2) the sample size, which is used as the denominator for the gradient\n        3) logging outputs to display while training\n        '
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Aggregate logging outputs from data parallel training.'
        utils.deprecation_warning('The aggregate_logging_outputs API is deprecated. Please use the reduce_metrics API instead.')
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        if False:
            i = 10
            return i + 15
        'Aggregate logging outputs from data parallel training.'
        utils.deprecation_warning('Criterions should implement the reduce_metrics API. Falling back to deprecated aggregate_logging_outputs API.')
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for (k, v) in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Whether the logging outputs returned by `forward` can be summed\n        across workers prior to calling `reduce_metrics`. Setting this\n        to True will improves distributed training speed.\n        '
        return False

class LegacyFairseqCriterion(FairseqCriterion):

    def __init__(self, args, task):
        if False:
            print('Hello World!')
        super().__init__(task=task)
        self.args = args
        utils.deprecation_warning('Criterions should take explicit arguments instead of an argparse.Namespace object, please update your criterion by extending FairseqCriterion instead of LegacyFairseqCriterion.')

    @classmethod
    def build_criterion(cls, args, task):
        if False:
            while True:
                i = 10
        'Construct a criterion from command-line args.'
        return cls(args, task)