from with_wandb import defs

def test_defs_can_load():
    if False:
        return 10
    assert defs.get_job_def('simple_job_example')
    assert defs.get_job_def('partitioned_job_example')
    assert defs.get_job_def('run_launch_agent_example')
    assert defs.get_job_def('run_launch_job_example')