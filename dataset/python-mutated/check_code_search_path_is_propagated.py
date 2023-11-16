"""
A dummy ray driver script that executes in subprocess.
Prints global worker's `load_code_from_local` property that ought to be set
whenever `JobConfig.code_search_path` is specified
"""

def run():
    if False:
        return 10
    import ray
    from ray.job_config import JobConfig
    ray.init(job_config=JobConfig(code_search_path=['/home/code/']))

    @ray.remote
    def foo() -> bool:
        if False:
            while True:
                i = 10
        return ray._private.worker.global_worker.load_code_from_local
    load_code_from_local = ray.get(foo.remote())
    statement = 'propagated' if load_code_from_local else 'NOT propagated'
    print(f'Code search path is {statement}')
    print(ray.get_runtime_context().runtime_env)
if __name__ == '__main__':
    run()