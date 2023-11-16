"""Test that compute log tail processes go away when the parent hard crashes."""

import sys

from dagster._core.execution.compute_logs import mirror_stream_to_file
from dagster._utils import segfault

if __name__ == "__main__":
    stdout_pids_file, stderr_pids_file = (sys.argv[1], sys.argv[2])
    with mirror_stream_to_file(sys.stdout, stdout_pids_file) as stdout_pids:
        with mirror_stream_to_file(sys.stderr, stderr_pids_file) as stderr_pids:
            sys.stdout.write(f"stdout pids: {stdout_pids}")
            sys.stdout.flush()
            sys.stderr.write(f"stderr pids: {stderr_pids}")
            sys.stderr.flush()
            segfault()
