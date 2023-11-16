import argparse
import subprocess
import sys
from typing import List, Optional
parser = argparse.ArgumentParser()
parser.add_argument('-q', '--quiet', action='count')
parser.add_argument('packages', type=str, nargs='*', help='Additional packages (with optional version reqs) to pass to `pip install`')
parser.add_argument('--include-prebuilt-grpcio-wheel', action='store_true')

def main(quiet: bool, extra_packages: List[str], include_prebuilt_grpcio_wheel: Optional[bool]) -> None:
    if False:
        while True:
            i = 10
    'Especially on macOS, there may be missing wheels for new major Python versions, which means that\n    some dependencies may have to be built from source. You may find yourself needing to install\n    system packages such as freetype, gfortran, etc.; on macOS, Homebrew should suffice.\n    '
    install_targets: List[str] = [*extra_packages]
    install_targets += ['-e python_modules/dagster[pyright,ruff,test]', '-e python_modules/dagster-pipes', '-e python_modules/dagster-graphql', '-e python_modules/dagster-test', '-e python_modules/dagster-webserver', '-e python_modules/dagit', '-e python_modules/automation', '-e python_modules/libraries/dagster-managed-elements', '-e python_modules/libraries/dagster-airbyte', '-e python_modules/libraries/dagster-airflow', '-e python_modules/libraries/dagster-aws[test]', '-e python_modules/libraries/dagster-celery', '-e python_modules/libraries/dagster-celery-docker', '-e python_modules/libraries/dagster-dask[yarn,pbs,kube]', '-e python_modules/libraries/dagster-databricks', '-e python_modules/libraries/dagster-datadog', '-e python_modules/libraries/dagster-datahub', '-e python_modules/libraries/dagster-dbt', '-e python_modules/libraries/dagster-docker', '-e python_modules/libraries/dagster-gcp', '-e python_modules/libraries/dagster-gcp-pandas', '-e python_modules/libraries/dagster-gcp-pyspark', '-e python_modules/libraries/dagster-embedded-elt', '-e python_modules/libraries/dagster-fivetran', '-e python_modules/libraries/dagster-k8s', '-e python_modules/libraries/dagster-celery-k8s', '-e python_modules/libraries/dagster-github', '-e python_modules/libraries/dagster-mlflow', '-e python_modules/libraries/dagster-mysql', '-e python_modules/libraries/dagster-pagerduty', '-e python_modules/libraries/dagster-pandas', '-e python_modules/libraries/dagster-papertrail', '-e python_modules/libraries/dagster-postgres', '-e python_modules/libraries/dagster-prometheus', '-e python_modules/libraries/dagster-pyspark', '-e python_modules/libraries/dagster-shell', '-e python_modules/libraries/dagster-slack', '-e python_modules/libraries/dagster-spark', '-e python_modules/libraries/dagster-ssh', '-e python_modules/libraries/dagster-twilio', '-e python_modules/libraries/dagstermill', '-e integration_tests/python_modules/dagster-k8s-test-infra', '-e python_modules/libraries/dagster-azure', '-e python_modules/libraries/dagster-msteams', '-e python_modules/libraries/dagster-duckdb', '-e python_modules/libraries/dagster-duckdb-pandas', '-e python_modules/libraries/dagster-duckdb-polars', '-e python_modules/libraries/dagster-duckdb-pyspark', '-e python_modules/libraries/dagster-wandb', '-e python_modules/libraries/dagster-deltalake', '-e python_modules/libraries/dagster-deltalake-pandas', '-e python_modules/libraries/dagster-deltalake-polars', '-e helm/dagster/schema[test]', '-e .buildkite/dagster-buildkite']
    if sys.version_info > (3, 7):
        install_targets += ['-e python_modules/libraries/dagster-dbt', '-e python_modules/libraries/dagster-pandera', '-e python_modules/libraries/dagster-snowflake', '-e python_modules/libraries/dagster-snowflake-pandas']
    if sys.version_info > (3, 6) and sys.version_info < (3, 10):
        install_targets += []
    if include_prebuilt_grpcio_wheel:
        install_targets += ['--find-links', 'https://github.com/dagster-io/build-grpcio/wiki/Wheels']
    cmd = ['pip', 'install'] + install_targets
    if quiet is not None:
        cmd.append(f"-{'q' * quiet}")
    p = subprocess.Popen(' '.join(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    print(' '.join(cmd))
    while True:
        output = p.stdout.readline()
        if p.poll() is not None:
            break
        if output:
            print(output.decode('utf-8').strip())
if __name__ == '__main__':
    args = parser.parse_args()
    main(quiet=args.quiet, extra_packages=args.packages, include_prebuilt_grpcio_wheel=args.include_prebuilt_grpcio_wheel)