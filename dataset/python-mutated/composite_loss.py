from fairseq import utils
from fairseq.criterions import LegacyFairseqCriterion, register_criterion
from torch import nn

@register_criterion('composite_loss')
class CompositeLoss(LegacyFairseqCriterion):
    """This is a composite loss that, given a list of model outputs and a list of targets,
    computes an average of losses for each output-target pair"""

    def __init__(self, args, task):
        if False:
            while True:
                i = 10
        super().__init__(args, task)
        self.underlying_criterion = args.underlying_criterion

    @staticmethod
    def add_args(parser):
        if False:
            for i in range(10):
                print('nop')
        'Add criterion-specific arguments to the parser.'
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True, help='underlying criterion to use for the composite loss')

    @staticmethod
    def build_underlying_criterion(args, task):
        if False:
            print('Hello World!')
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion
        assert saved_criterion != args.underlying_criterion
        underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        return underlying_criterion

    @classmethod
    def build_criterion(cls, args, task):
        if False:
            while True:
                i = 10
        underlying_criterion = CompositeLoss.build_underlying_criterion(args, task)

        class FakeModel(nn.Module):

            def __init__(self, model, net_out, target):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.model = model
                self.net_out = net_out
                self.target = target

            def forward(self, **unused):
                if False:
                    return 10
                return self.net_out

            def get_normalized_probs(self, net_output, log_probs, sample=None):
                if False:
                    i = 10
                    return i + 15
                return self.model.get_normalized_probs(net_output, log_probs, sample=sample)

            def get_targets(self, *unused):
                if False:
                    for i in range(10):
                        print('nop')
                return self.target

            @property
            def decoder(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.model.decoder

        class _CompositeLoss(LegacyFairseqCriterion):

            def __init__(self, args, task, underlying_criterion):
                if False:
                    return 10
                super().__init__(args, task)
                self.underlying_criterion = underlying_criterion

            def forward(self, model, sample, reduce=True):
                if False:
                    i = 10
                    return i + 15
                net_outputs = model(**sample['net_input'])
                targets = sample['target']
                bsz = targets[0].size(0)
                loss = net_outputs[0][0].new(1 if reduce else bsz).float().zero_()
                sample_size = 0
                logging_output = {}
                for (o, t) in zip(net_outputs[0], targets):
                    m = FakeModel(model, (o, net_outputs[1]), t)
                    sample['target'] = t
                    (l, ss, logging_output) = self.underlying_criterion(m, sample, reduce)
                    loss += l
                    sample_size += ss
                loss.div_(len(targets))
                sample_size /= len(targets)
                logging_output['loss'] = utils.item(loss.data) if reduce else loss.data
                return (loss, sample_size, logging_output)

            @staticmethod
            def aggregate_logging_outputs(logging_outputs):
                if False:
                    i = 10
                    return i + 15
                return underlying_criterion.__class__.aggregate_logging_outputs(logging_outputs)

            @staticmethod
            def reduce_metrics(logging_outputs) -> None:
                if False:
                    i = 10
                    return i + 15
                underlying_criterion.__class__.reduce_metrics(logging_outputs)
        return _CompositeLoss(args, task, underlying_criterion)