import os
import ray

def run():
    if False:
        i = 10
        return i + 15
    ray.init()

    @ray.remote(runtime_env={'env_vars': {'FOO': 'bar'}})
    def get_task_working_dir():
        if False:
            while True:
                i = 10
        assert os.path.exists('per_task_runtime_env.py')
        return ray.get_runtime_context().runtime_env.working_dir()
    driver_working_dir = ray.get_runtime_context().runtime_env.working_dir()
    task_working_dir = ray.get(get_task_working_dir.remote())
    assert driver_working_dir == task_working_dir, (driver_working_dir, task_working_dir)
if __name__ == '__main__':
    run()