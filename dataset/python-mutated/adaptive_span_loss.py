import math
from dataclasses import dataclass
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class AdaptiveSpanCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II('optimization.sentence_avg')

@register_criterion('adaptive_span_loss', dataclass=AdaptiveSpanCriterionConfig)
class AdaptiveSpanCriterion(CrossEntropyCriterion):

    def __init__(self, task, sentence_avg):
        if False:
            print('Hello World!')
        super().__init__(task, sentence_avg)

    def forward(self, model, sample, reduce=True):
        if False:
            for i in range(10):
                print('nop')
        'Compute the loss for the given sample.\n\n        Returns a tuple with three elements:\n        1) the loss here is summed, different from the adaptive span code\n        2) the sample size, which is used as the denominator for the gradient\n        3) logging outputs to display while training\n        '
        net_output = model(**sample['net_input'])
        (loss, aux_loss, avg_span, max_span) = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        loss /= sample_size
        total_loss = loss + aux_loss
        sample_size = 1
        logging_output = {'loss': loss.data, 'ntokens': sample['ntokens'], 'nsentences': sample['target'].size(0), 'sample_size': sample_size, 'total_loss': total_loss.data, 'avg_span': avg_span * sample_size, 'max_span': max_span * sample_size}
        return (total_loss, sample_size, logging_output)

    def compute_loss(self, model, net_output, sample, reduce=True):
        if False:
            return 10
        (loss, _) = super().compute_loss(model, net_output, sample, reduce)
        aux_loss = model.get_aux_loss()
        avg_span = model.get_current_avg_span()
        max_span = model.get_current_max_span()
        return (loss, aux_loss, avg_span, max_span)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        if False:
            while True:
                i = 10
        'Aggregate logging outputs from data parallel training.'
        loss_sum = sum((log.get('loss', 0) for log in logging_outputs))
        ntokens = sum((log.get('ntokens', 0) for log in logging_outputs))
        sample_size = sum((log.get('sample_size', 0) for log in logging_outputs))
        total_loss_sum = sum((log.get('total_loss', 0) for log in logging_outputs))
        avg_span_sum = sum((log.get('avg_span', 0) for log in logging_outputs))
        max_span_sum = sum((log.get('max_span', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('avg_span', avg_span_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('max_span', max_span_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('total_loss', total_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        if False:
            return 10
        '\n        Whether the logging outputs returned by `forward` can be summed\n        across workers prior to calling `reduce_metrics`. Setting this\n        to True will improves distributed training speed.\n        '
        return True