import unittest
import test_pipeline

class TestPipelineWithIRPass(test_pipeline.TestPipeline):

    def need_envs(self):
        if False:
            return 10
        return {'FLAGS_apply_pass_to_program': '1', 'FLAGS_new_executor_micro_batching': '0'}
if __name__ == '__main__':
    unittest.main()