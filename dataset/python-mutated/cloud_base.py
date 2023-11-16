"""This module contains a Google API base operator."""
from __future__ import annotations
from google.api_core.gapic_v1.method import DEFAULT
from airflow.models import BaseOperator

class GoogleCloudBaseOperator(BaseOperator):
    """Abstract base class for operators using Google API client libraries."""

    def __deepcopy__(self, memo):
        if False:
            return 10
        '\n        Updating the memo to fix the non-copyable global constant.\n\n        This constant can be specified in operator parameters as a retry configuration to indicate a default.\n        See https://github.com/apache/airflow/issues/28751 for details.\n        '
        memo[id(DEFAULT)] = DEFAULT
        return super().__deepcopy__(memo)