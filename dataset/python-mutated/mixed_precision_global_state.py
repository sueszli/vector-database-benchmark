"""Contains global variables related to mixed precision.

This is not part of mixed_precision.py to avoid a circular dependency.
mixed_precision.py depends on Session, and Session depends on this file.
"""
from tensorflow.python.util.tf_export import tf_export
_mixed_precision_graph_rewrite_is_enabled = False
_non_mixed_precision_session_created = False
_using_mixed_precision_policy = False

@tf_export('__internal__.train.is_mixed_precision_graph_rewrite_enabled', v1=[])
def is_mixed_precision_graph_rewrite_enabled():
    if False:
        while True:
            i = 10
    return _mixed_precision_graph_rewrite_is_enabled

def set_mixed_precision_graph_rewrite_enabled(enabled):
    if False:
        return 10
    global _mixed_precision_graph_rewrite_is_enabled
    _mixed_precision_graph_rewrite_is_enabled = enabled

def non_mixed_precision_session_created():
    if False:
        return 10
    return _non_mixed_precision_session_created

def set_non_mixed_precision_session_created(created):
    if False:
        for i in range(10):
            print('nop')
    global _non_mixed_precision_session_created
    _non_mixed_precision_session_created = created

def is_using_mixed_precision_policy():
    if False:
        i = 10
        return i + 15
    return _using_mixed_precision_policy

@tf_export('__internal__.train.set_using_mixed_precision_policy', v1=[])
def set_using_mixed_precision_policy(is_using):
    if False:
        print('Hello World!')
    global _using_mixed_precision_policy
    _using_mixed_precision_policy = is_using