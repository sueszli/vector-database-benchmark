from __future__ import unicode_literals
import pytest
from dagster import job
from dagster._core.errors import DagsterInvariantViolationError
from dagster._core.execution.plan.plan import ExecutionPlan
from dagster._core.snap.execution_plan_snapshot import ExecutionPlanSnapshot
from dagster._serdes.serdes import deserialize_value
OLD_EXECUTION_PLAN_SNAPSHOT = '{\n  "__class__": "ExecutionPlanSnapshot",\n  "artifacts_persisted": true,\n  "initial_known_state": null,\n  "pipeline_snapshot_id": "0baebbaa257331ddeaccd3076c3dca413e099cf4",\n  "step_keys_to_execute": [\n    "solid_one",\n    "solid_two"\n  ],\n  "steps": [\n    {\n      "__class__": "ExecutionStepSnap",\n      "inputs": [],\n      "key": "solid_one",\n      "kind": {\n        "__enum__": "StepKind.COMPUTE"\n      },\n      "metadata_items": [],\n      "outputs": [\n        {\n          "__class__": "ExecutionStepOutputSnap",\n          "dagster_type_key": "Any",\n          "name": "result",\n          "properties": {\n            "__class__": "StepOutputProperties",\n            "is_asset": false,\n            "is_dynamic": false,\n            "is_required": true,\n            "should_materialize": false\n          },\n          "solid_handle": {\n            "__class__": "SolidHandle",\n            "name": "solid_one",\n            "parent": null\n          }\n        }\n      ],\n      "solid_handle_id": "solid_one",\n      "step_handle": {\n        "__class__": "StepHandle",\n        "solid_handle": {\n          "__class__": "SolidHandle",\n          "name": "solid_one",\n          "parent": null\n        }\n      },\n      "tags": {}\n    },\n    {\n      "__class__": "ExecutionStepSnap",\n      "inputs": [\n        {\n          "__class__": "ExecutionStepInputSnap",\n          "dagster_type_key": "Any",\n          "name": "num",\n          "source": {\n            "__class__": "FromStepOutput",\n            "fan_in": false,\n            "input_name": "num",\n            "solid_handle": {\n              "__class__": "SolidHandle",\n              "name": "solid_two",\n              "parent": null\n            },\n            "step_output_handle": {\n              "__class__": "StepOutputHandle",\n              "mapping_key": null,\n              "output_name": "result",\n              "step_key": "solid_one"\n            }\n          },\n          "upstream_output_handles": [\n            {\n              "__class__": "StepOutputHandle",\n              "mapping_key": null,\n              "output_name": "result",\n              "step_key": "solid_one"\n            }\n          ]\n        }\n      ],\n      "key": "solid_two",\n      "kind": {\n        "__enum__": "StepKind.COMPUTE"\n      },\n      "metadata_items": [],\n      "outputs": [\n        {\n          "__class__": "ExecutionStepOutputSnap",\n          "dagster_type_key": "Any",\n          "name": "result",\n          "properties": {\n            "__class__": "StepOutputProperties",\n            "is_asset": false,\n            "is_dynamic": false,\n            "is_required": true,\n            "should_materialize": false\n          },\n          "solid_handle": {\n            "__class__": "SolidHandle",\n            "name": "solid_two",\n            "parent": null\n          }\n        }\n      ],\n      "solid_handle_id": "solid_two",\n      "step_handle": {\n        "__class__": "StepHandle",\n        "solid_handle": {\n          "__class__": "SolidHandle",\n          "name": "solid_two",\n          "parent": null\n        }\n      },\n      "tags": {}\n    }\n  ]\n}'

@job
def noop_job():
    if False:
        while True:
            i = 10
    pass

def test_cant_load_old_snapshot():
    if False:
        i = 10
        return i + 15
    snapshot = deserialize_value(OLD_EXECUTION_PLAN_SNAPSHOT, ExecutionPlanSnapshot)
    with pytest.raises(DagsterInvariantViolationError, match='Tried to reconstruct an old ExecutionPlanSnapshot that was created before snapshots had enough information to fully reconstruct the ExecutionPlan'):
        ExecutionPlan.rebuild_from_snapshot('noop_job', snapshot)
PRE_CACHE_EXECUTION_PLAN_SNAPSHOT = '{\n  "__class__": "ExecutionPlanSnapshot",\n  "artifacts_persisted": true,\n  "initial_known_state": null,\n  "pipeline_snapshot_id": "0965b76124e758660317760c7e9bbc66282f33b0",\n  "snapshot_version": 1,\n  "step_keys_to_execute": [\n    "noop_solid"\n  ],\n  "step_output_versions": [],\n  "steps": [\n    {\n      "__class__": "ExecutionStepSnap",\n      "inputs": [],\n      "key": "noop_solid",\n      "kind": {\n        "__enum__": "StepKind.COMPUTE"\n      },\n      "metadata_items": [],\n      "outputs": [\n        {\n          "__class__": "ExecutionStepOutputSnap",\n          "dagster_type_key": "Any",\n          "name": "result",\n          "properties": {\n            "__class__": "StepOutputProperties",\n            "is_asset": false,\n            "is_dynamic": false,\n            "is_required": true,\n            "should_materialize": false\n          },\n          "solid_handle": {\n            "__class__": "SolidHandle",\n            "name": "noop_solid",\n            "parent": null\n          }\n        }\n      ],\n      "solid_handle_id": "noop_solid",\n      "step_handle": {\n        "__class__": "StepHandle",\n        "solid_handle": {\n          "__class__": "SolidHandle",\n          "name": "noop_solid",\n          "parent": null\n        }\n      },\n      "tags": {}\n    }\n  ]\n}'

def test_rebuild_pre_cached_key_execution_plan_snapshot():
    if False:
        for i in range(10):
            print('nop')
    snapshot = deserialize_value(PRE_CACHE_EXECUTION_PLAN_SNAPSHOT, ExecutionPlanSnapshot)
    ExecutionPlan.rebuild_from_snapshot('noop_job', snapshot)