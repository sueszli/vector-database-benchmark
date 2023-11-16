import unittest
from paddle import static
from paddle.base import core

class TestStandaloneExecutorPlan(unittest.TestCase):

    def test_standalone_executor_plan(self):
        if False:
            for i in range(10):
                print('nop')
        micro_batch_id = 0
        forward_job = core.Job('forward')
        backward_job = core.Job('backward')
        optimizer_job = core.Job('optimizer')
        forward_job.set_micro_batch_id(micro_batch_id)
        backward_job.set_micro_batch_id(micro_batch_id)
        optimizer_job.set_micro_batch_id(micro_batch_id)
        self.assertEqual(forward_job.micro_batch_id(), micro_batch_id)
        self.assertEqual(forward_job.type(), 'forward')
        forward_program = static.Program()
        backward_program = static.Program()
        optimizer_program = static.Program()
        job_list = [forward_job, backward_job, optimizer_job]
        type_to_program = {'forward': forward_program.desc, 'backward': backward_program.desc, 'optimizer': optimizer_program.desc}
        plan = core.Plan(job_list, type_to_program)
        self.assertEqual(plan.job_list(), job_list)
        for type in type_to_program.keys():
            self.assertEqual(plan.program(type), type_to_program[type])
if __name__ == '__main__':
    unittest.main()