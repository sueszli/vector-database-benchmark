import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef

def _resolve_workflow_refs(index: int) -> Any:
    if False:
        i = 10
        return i + 15
    raise ValueError('There is no context for resolving workflow refs.')

@contextlib.contextmanager
def workflow_args_serialization_context(workflow_refs: List[WorkflowRef]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    This serialization context reduces workflow input arguments to three\n    parts:\n\n    1. A workflow input placeholder. It is an object without \'Workflow\' and\n       \'ObjectRef\' object. They are replaced with integer indices. During\n       deserialization, we can refill the placeholder with a list of\n       \'Workflow\' and a list of \'ObjectRef\'. This provides us great\n       flexibility, for example, during recovery we can plug an alternative\n       list of \'Workflow\' and \'ObjectRef\', since we lose the original ones.\n    2. A list of \'Workflow\'. There is no duplication in it.\n    3. A list of \'ObjectRef\'. There is no duplication in it.\n\n    We do not allow duplication because in the arguments duplicated workflows\n    and object refs are shared by reference. So when deserialized, we also\n    want them to be shared by reference. See\n    "tests/test_object_deref.py:deref_shared" as an example.\n\n    The deduplication works like this:\n        Inputs: [A B A B C C A]\n        Output List: [A B C]\n        Index in placeholder: [0 1 0 1 2 2 0]\n\n    Args:\n        workflow_refs: Output list of workflows or references to workflows.\n    '
    deduplicator: Dict[WorkflowRef, int] = {}

    def serializer(w):
        if False:
            i = 10
            return i + 15
        if w in deduplicator:
            return deduplicator[w]
        if isinstance(w, WorkflowRef):
            w.ref = None
        i = len(workflow_refs)
        workflow_refs.append(w)
        deduplicator[w] = i
        return i
    register_serializer(WorkflowRef, serializer=serializer, deserializer=_resolve_workflow_refs)
    try:
        yield
    finally:
        deregister_serializer(WorkflowRef)

@contextlib.contextmanager
def workflow_args_resolving_context(workflow_ref_mapping: List[Any]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    This context resolves workflows and object refs inside workflow\n    arguments into correct values.\n\n    Args:\n        workflow_ref_mapping: List of workflow refs.\n    '
    global _resolve_workflow_refs
    _resolve_workflow_refs_bak = _resolve_workflow_refs
    _resolve_workflow_refs = workflow_ref_mapping.__getitem__
    try:
        yield
    finally:
        _resolve_workflow_refs = _resolve_workflow_refs_bak

class _KeepWorkflowRefs:

    def __init__(self, index: int):
        if False:
            i = 10
            return i + 15
        self._index = index

    def __reduce__(self):
        if False:
            return 10
        return (_resolve_workflow_refs, (self._index,))

@contextlib.contextmanager
def workflow_args_keeping_context() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    This context only read workflow arguments. Workflows inside\n    are untouched and can be serialized again properly.\n    '
    global _resolve_workflow_refs
    _resolve_workflow_refs_bak = _resolve_workflow_refs

    def _keep_workflow_refs(index: int):
        if False:
            i = 10
            return i + 15
        return _KeepWorkflowRefs(index)
    _resolve_workflow_refs = _keep_workflow_refs
    try:
        yield
    finally:
        _resolve_workflow_refs = _resolve_workflow_refs_bak