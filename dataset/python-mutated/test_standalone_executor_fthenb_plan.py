import unittest
from paddle import static
from paddle.distributed.passes import PassContext, new_pass

class TestStandaloneExecutorFThenBPlan(unittest.TestCase):

    def test_standalone_executor_fthenb_plan(self):
        if False:
            for i in range(10):
                print('nop')
        config = {}
        config['num_micro_batches'] = 4
        pass_context = PassContext()
        startup_program = static.Program()
        main_program = static.Program()
        pipeline_fthenb_pass = new_pass('pipeline_scheduler_FThenB', config)
        pipeline_fthenb_pass.apply([main_program], [startup_program], pass_context)
        plan = pass_context.get_attr('plan')
        job_type_list = []
        for job in plan.job_list():
            job_type_list.append(job.type())
        expect_job_type_list = ['forward', 'forward', 'forward', 'forward', 'backward', 'backward', 'backward', 'backward', 'optimizer']
        self.assertEqual(job_type_list, expect_job_type_list)
if __name__ == '__main__':
    unittest.main()