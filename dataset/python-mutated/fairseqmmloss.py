"""
TODO (huxu): a general fairseq criterion for all your pre-defined losses.
"""
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics

@register_criterion('mmloss')
class MMCriterion(FairseqCriterion):

    def __init__(self, task):
        if False:
            i = 10
            return i + 15
        super().__init__(task)
        self.mmtask = task.mmtask

    def forward(self, model, sample):
        if False:
            return 10
        'Compute the loss for the given sample.\n        Returns a tuple with three elements:\n        1) the loss\n        2) the sample size, which is used as the denominator for the gradient\n        3) logging outputs to display while training\n        '
        outputs = self.mmtask(model, sample)
        (loss, loss_scalar, max_len, batch_size, sample_size) = (outputs['loss'], outputs['loss_scalar'], outputs['max_len'], outputs['batch_size'], outputs['sample_size'])
        logging_output = {'loss': loss_scalar, 'ntokens': max_len * batch_size, 'nsentences': batch_size, 'sample_size': sample_size}
        return (loss, 1, logging_output)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Aggregate logging outputs from data parallel training.'
        'since we use NCE, our actual batch_size is 1 per GPU.\n        Then we take the mean of each worker.'
        loss_sum = sum((log.get('loss', 0.0) for log in logging_outputs))
        sample_size = sum((log.get('sample_size', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss_sum / sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether the logging outputs returned by `forward` can be summed\n        across workers prior to calling `reduce_metrics`. Setting this\n        to True will improves distributed training speed.\n        '
        return True