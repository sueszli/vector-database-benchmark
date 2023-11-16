"""Provides a seam for transaction services."""
from __future__ import annotations
import functools
from google.cloud import datastore
from typing import Any, Callable
CLIENT = datastore.Client()

def run_in_transaction_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    if False:
        return 10
    'Runs a decorated function in a transaction. Either all of the operations\n    in the transaction are applied, or none of them are applied.\n\n    If an exception is raised, the transaction is likely not safe to\n    commit, since TransactionOptions.ALLOWED is used.\n\n    Returns:\n        function. Function wrapped in transaction.\n\n    Raises:\n        Exception. Whatever fn() raises.\n        datastore_errors.TransactionFailedError. The transaction failed.\n    '

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if False:
            return 10
        'Wrapper for the transaction.'
        with CLIENT.transaction():
            return fn(*args, **kwargs)
    return wrapper