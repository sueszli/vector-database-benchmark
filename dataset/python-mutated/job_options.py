"""Option class for configuring the behavior of Oppia jobs."""
from __future__ import annotations
import argparse
from core import feconf
from apache_beam.options import pipeline_options
from typing import List, Optional

class JobOptions(pipeline_options.PipelineOptions):
    """Option class for configuring the behavior of Oppia jobs."""
    JOB_OPTIONS = {'namespace': (str, 'Namespace for isolating the NDB operations during tests.')}

    def __init__(self, flags: Optional[List[str]]=None, **job_options: Optional[str]) -> None:
        if False:
            while True:
                i = 10
        "Initializes a new JobOptions instance.\n\n        Args:\n            flags: list(str)|None. Command-line flags for customizing a\n                pipeline. Although Oppia doesn't use command-line flags to\n                control jobs or pipelines, we still need to pass the value\n                (unmodified) because PipelineOptions, a parent class, needs it.\n            **job_options: dict(str: *). One of the options defined in the class\n                JOB_OPTIONS dict.\n\n        Raises:\n            ValueError. Unsupported job option(s).\n        "
        unsupported_options = set(job_options).difference(self.JOB_OPTIONS)
        if unsupported_options:
            joined_unsupported_options = ', '.join(sorted(unsupported_options))
            raise ValueError('Unsupported option(s): %s' % joined_unsupported_options)
        super().__init__(flags=flags, project=feconf.OPPIA_PROJECT_ID, region=feconf.GOOGLE_APP_ENGINE_REGION, temp_location=feconf.DATAFLOW_TEMP_LOCATION, staging_location=feconf.DATAFLOW_STAGING_LOCATION, experiments=['use_runner_v2', 'enable_recommendations'], extra_packages=[feconf.OPPIA_PYTHON_PACKAGE_PATH], **job_options)

    @classmethod
    def _add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Adds Oppia's job-specific arguments to the parser.\n\n        Args:\n            parser: argparse.ArgumentParser. An ArgumentParser instance.\n        "
        for (option_name, (option_type, option_doc)) in cls.JOB_OPTIONS.items():
            parser.add_argument('--%s' % option_name, help=option_doc, type=option_type)