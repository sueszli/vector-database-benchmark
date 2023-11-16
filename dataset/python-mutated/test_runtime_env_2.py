import sys
import pytest
from ray._private.test_utils import run_string_as_driver

@pytest.mark.skipif(sys.platform == 'win32', reason='Fail to create temp dir.')
def test_working_dir_deploy_new_version(ray_start, tmp_dir):
    if False:
        return 10
    with open('hello', 'w') as f:
        f.write('world')
    driver1 = '\nimport ray\nfrom ray import serve\n\njob_config = ray.job_config.JobConfig(runtime_env={"working_dir": "."})\nray.init(address="auto", namespace="serve", job_config=job_config)\n\n\n@serve.deployment(version="1")\nclass Test:\n    def __call__(self, *args):\n        return open("hello").read()\n\nhandle = serve.run(Test.bind())\nassert handle.remote().result() == "world"\n'
    run_string_as_driver(driver1)
    with open('hello', 'w') as f:
        f.write('world2')
    driver2 = '\nimport ray\nfrom ray import serve\nfrom ray.serve._private.constants import SERVE_DEFAULT_APP_NAME\n\njob_config = ray.job_config.JobConfig(runtime_env={"working_dir": "."})\nray.init(address="auto", namespace="serve", job_config=job_config)\n\n\n@serve.deployment(version="2")\nclass Test:\n    def __call__(self, *args):\n        return open("hello").read()\n\nhandle = serve.run(Test.bind())\nassert handle.remote().result() == "world2"\nserve.delete(SERVE_DEFAULT_APP_NAME)\n'
    run_string_as_driver(driver2)

@pytest.mark.skipif(sys.platform == 'win32', reason='Runtime env unsupported on Windows')
def test_pip_no_working_dir(ray_start):
    if False:
        return 10
    driver = '\nimport ray\nfrom ray import serve\nimport requests\n\nray.init(address="auto")\n\n\n@serve.deployment\ndef requests_version(request):\n    return requests.__version__\n\n\nserve.run(requests_version.options(\n    ray_actor_options={\n        "runtime_env": {\n            "pip": ["requests==2.25.1"]\n        }\n    }).bind())\n\nassert requests.get("http://127.0.0.1:8000/requests_version").text == "2.25.1"\n'
    output = run_string_as_driver(driver)
    print(output)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-sv', __file__]))