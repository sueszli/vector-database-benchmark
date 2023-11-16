import os
import sys
import tempfile
from io import BytesIO

import yaml
from dagster import execute_job, reconstructable
from dagster._cli.job import execute_list_versions_command
from dagster._core.test_utils import instance_for_test
from dagster._utils import file_relative_path

from ...execution_tests.memoized_dev_loop_job import op_job


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stringio = BytesIO()
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def test_execute_display_command():
    with tempfile.TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            run_config = {
                "ops": {
                    "create_string_1_asset_op": {"config": {"input_str": "apple"}},
                    "take_string_1_asset_op": {"config": {"input_str": "apple"}},
                },
                "resources": {"io_manager": {"config": {"base_dir": temp_dir}}},
            }

            # write run config to temp file
            # file is temp because io manager directory is temporary
            with open(os.path.join(temp_dir, "job_config.yaml"), "w", encoding="utf8") as f:
                f.write(yaml.dump(run_config))

            kwargs = {
                "config": (os.path.join(temp_dir, "job_config.yaml"),),
                "job_name": "op_job",
                "python_file": file_relative_path(
                    __file__, "../../execution_tests/memoized_dev_loop_job.py"
                ),
                "tags": '{"dagster/is_memoized_run": "true"}',
            }

            with Capturing() as output:
                execute_list_versions_command(kwargs=kwargs, instance=instance)

            assert output

            # execute the pipeline once so that addresses have been populated.

            result = execute_job(reconstructable(op_job), instance=instance, run_config=run_config)
            assert result.success

            with Capturing() as output:
                execute_list_versions_command(kwargs=kwargs, instance=instance)

            assert output
