# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
run_pytest.py is the script submitted to Azure ML that runs pytest.
pytest runs all tests in the specified test folder unless parameters
are set otherwise.
"""

import sys
import logging
import pytest
import argparse
import glob
import pkg_resources
from azureml.core import Run
from test_groups import nightly_test_groups, pr_gate_test_groups

if __name__ == "__main__":

    logger = logging.getLogger("submit_groupwise_azureml_pytest.py")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument(
        "--testkind",
        "-k",
        action="store",
        default="unit",
        help="Test kind - nightly or unit",
    )
    parser.add_argument(
        "--testgroup",
        "-g",
        action="store",
        default="group_cpu_001",
        help="Group name for the tests",
    )
    # Flag to indicate whether to turn off the warnings
    parser.add_argument(
        "--disable-warnings",
        action="store_true",
        help="Turn off warnings",
    )
    args = parser.parse_args()

    if args.testkind == "nightly":
        test_group = nightly_test_groups[args.testgroup]
    else:
        test_group = pr_gate_test_groups[args.testgroup]

    logger.info(f"Python version: {sys.version}")

    logger.info("Installed packages:")
    for p in pkg_resources.working_set:
        logger.info(f" {p.project_name}:{p.version}")

    logger.info("Tests to be executed")
    logger.info(str(test_group))

    # Run.get_context() is needed to save context as pytest causes corruption
    # of env vars
    run = Run.get_context()

    logger.info("Executing tests now...")

    # Add options to pytest command (Duration and disable warnings)
    pytest_string = test_group + ["--durations"] + ["0"]
    if args.disable_warnings is True:
        pytest_string += ["--disable-warnings"]

    # Execute pytest command
    pytest_exit_code = pytest.main(pytest_string)

    logger.info("Test execution completed!")

    # log pytest exit code as a metric
    # to be used to indicate success/failure in github workflow
    run.log("pytest_exit_code", pytest_exit_code.value)

    # #
    # # Leveraged code from this  notebook:
    # # https://msdata.visualstudio.com/Vienna/_search?action=contents&text=upload_folder&type=code&lp=code-Project&filters=ProjectFilters%7BVienna%7DRepositoryFilters%7BAzureMlCli%7D&pageSize=25&sortOptions=%5B%7B%22field%22%3A%22relevance%22%2C%22sortOrder%22%3A%22desc%22%7D%5D&result=DefaultCollection%2FVienna%2FAzureMlCli%2FGBmaster%2F%2Fsrc%2Fazureml-core%2Fazureml%2Fcore%2Frun.py
    # logger.info("os.listdir files {}".format(os.listdir(".")))

    # upload pytest stdout file
    logs_path = (
        glob.glob("**/70_driver_log.txt", recursive=True)
        + glob.glob("**/std_log.txt", recursive=True)
    )[0]
    run.upload_file(name="test_logs", path_or_stream=logs_path)
