import unittest
from modelscope import TrainingArgs
from modelscope.trainers.cli_argument_parser import CliArgumentParser
from modelscope.utils.test_utils import test_level

class TrainingArgsTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        print('Testing %s.%s' % (type(self).__name__, self._testMethodName))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_define_args(self):
        if False:
            return 10
        myparser = CliArgumentParser(TrainingArgs())
        input_args = ['--max_epochs', '100', '--work_dir', 'ddddd', '--per_device_train_batch_size', '8', '--unkown', 'unkown']
        (args, remainning) = myparser.parse_known_args(input_args)
        myparser.print_help()
        self.assertTrue(args.max_epochs == 100)
        self.assertTrue(args.work_dir == 'ddddd')
        self.assertTrue(args.per_device_train_batch_size == 8)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_flatten_args(self):
        if False:
            i = 10
            return i + 15
        training_args = TrainingArgs()
        input_args = ['--optimizer_params', 'weight_decay=0.8,eps=1e-6,correct_bias=False', '--lr_scheduler_params', 'initial_lr=3e-5,niter_decay=1']
        training_args = training_args.parse_cli(input_args)
        (cfg, _) = training_args.to_config()
        self.assertAlmostEqual(cfg.train.optimizer.weight_decay, 0.8)
        self.assertAlmostEqual(cfg.train.optimizer.eps, 1e-06)
        self.assertFalse(cfg.train.optimizer.correct_bias)
        self.assertAlmostEqual(cfg.train.lr_scheduler.initial_lr, 3e-05)
        self.assertEqual(cfg.train.lr_scheduler.niter_decay, 1)
if __name__ == '__main__':
    unittest.main()