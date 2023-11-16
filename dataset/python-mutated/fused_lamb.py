from fairseq.optim import LegacyFairseqOptimizer, register_optimizer

@register_optimizer('lamb')
class FairseqLAMB(LegacyFairseqOptimizer):
    """LAMB optimizer."""

    def __init__(self, args, params):
        if False:
            i = 10
            return i + 15
        super().__init__(args)
        try:
            from apex.optimizers import FusedLAMB
            self._optimizer = FusedLAMB(params, **self.optimizer_config)
        except ImportError:
            raise ImportError('Please install apex to use LAMB optimizer')

    @staticmethod
    def add_args(parser):
        if False:
            i = 10
            return i + 15
        'Add optimizer-specific arguments to the parser.'
        parser.add_argument('--lamb-betas', default='(0.9, 0.999)', metavar='B', help='betas for LAMB optimizer')
        parser.add_argument('--lamb-eps', type=float, default=1e-08, metavar='D', help='epsilon for LAMB optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD', help='weight decay')

    @property
    def optimizer_config(self):
        if False:
            return 10
        '\n        Return a kwarg dictionary that will be used to override optimizer\n        args stored in checkpoints. This allows us to load a checkpoint and\n        resume training using a different set of optimizer args, e.g., with a\n        different learning rate.\n        '
        return {'lr': self.args.lr[0], 'betas': eval(self.args.lamb_betas), 'eps': self.args.lamb_eps, 'weight_decay': self.args.weight_decay}

    @property
    def supports_flat_params(self):
        if False:
            while True:
                i = 10
        return False