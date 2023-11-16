import sys
import pytest
import ray
from ray import serve
from ray._private.test_utils import run_string_as_driver

@pytest.mark.skipif(sys.platform == 'win32', reason='Fail to create temp dir.')
def test_failure_condition(ray_start, tmp_dir):
    if False:
        while True:
            i = 10
    with open('hello', 'w') as f:
        f.write('world')
    driver = '\nimport ray\nfrom ray import serve\n\nray.init(address="auto")\n\n\n@serve.deployment\nclass Test:\n    def __call__(self, *args):\n        return open("hello").read()\n\nhandle = serve.run(Test.bind())\ntry:\n    handle.remote().result()\n    assert False, "Should not get here"\nexcept FileNotFoundError:\n    pass\n'
    run_string_as_driver(driver)

@pytest.mark.skipif(sys.platform == 'win32', reason='Fail to create temp dir.')
def test_working_dir_basic(ray_start, tmp_dir, ray_shutdown):
    if False:
        for i in range(10):
            print('nop')
    with open('hello', 'w') as f:
        f.write('world')
    print('Wrote file')
    ray.init(address='auto', namespace='serve', runtime_env={'working_dir': '.'})
    print('Initialized Ray')

    @serve.deployment
    class Test:

        def __call__(self, *args):
            if False:
                i = 10
                return i + 15
            return open('hello').read()
    handle = serve.run(Test.bind())
    print('Deployed')
    assert handle.remote().result() == 'world'

@pytest.mark.skipif(sys.platform == 'win32', reason='Fail to create temp dir.')
def test_working_dir_connect_from_new_driver(ray_start, tmp_dir):
    if False:
        while True:
            i = 10
    with open('hello', 'w') as f:
        f.write('world')
    driver1 = '\nimport ray\nfrom ray import serve\n\njob_config = ray.job_config.JobConfig(runtime_env={"working_dir": "."})\nray.init(address="auto", namespace="serve", job_config=job_config)\n\n\n@serve.deployment\nclass Test:\n    def __call__(self, *args):\n        return open("hello").read()\n\nhandle = serve.run(Test.bind(), name="app")\nassert handle.remote().result() == "world"\n'
    run_string_as_driver(driver1)
    driver2 = driver1 + "\nserve.delete('app')"
    run_string_as_driver(driver2)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-sv', __file__]))