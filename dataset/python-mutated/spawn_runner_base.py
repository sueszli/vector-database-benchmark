import unittest
import numpy as np
from legacy_test.test_dist_base import RUN_STEP
import paddle

class SpawnAssistTestArgs:
    update_method = 'local'
    trainer_id = 0
    find_unused_parameters = False

class TestDistSpawnRunner(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.nprocs = 2

    def _run(self, model, args):
        if False:
            return 10
        args.update_method = 'local'
        return model.run_trainer_with_spawn(args)

    def _run_parallel(self, model, args):
        if False:
            i = 10
            return i + 15
        args.update_method = 'nccl2'
        context = paddle.distributed.spawn(func=model.run_trainer_with_spawn, args=(args,), nprocs=self.nprocs, join=True)
        result_list = []
        for res_queue in context.return_queues:
            result_list.append(res_queue.get())
        return result_list

    def check_dist_result_with_spawn(self, test_class, delta=0.001):
        if False:
            return 10
        self.check_dist_result_with_spawn_func(test_class=test_class, delta=delta)

    def check_dist_result_with_spawn_func(self, test_class, delta=0.001):
        if False:
            while True:
                i = 10
        model = test_class()
        args = SpawnAssistTestArgs()
        losses = self._run(model, args)
        dist_losses_list = self._run_parallel(model, args)
        for step_id in range(RUN_STEP):
            loss = losses[step_id]
            dist_loss_sum = None
            for dist_losses in dist_losses_list:
                if dist_loss_sum is None:
                    dist_loss_sum = np.array(dist_losses[step_id])
                else:
                    dist_loss_sum += np.array(dist_losses[step_id])
            dist_loss = dist_loss_sum / self.nprocs
            self.assertAlmostEqual(loss, dist_loss, delta=delta, msg=f'The results of single-card execution and multi-card execution are inconsistent.signal-card loss is:\n{loss}\nmulti-card average loss is:\n{dist_loss}\n')