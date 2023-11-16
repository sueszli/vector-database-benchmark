import dataclasses
import logging
from typing import Dict, List
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import SavePlan
__all__ = ['dedup_tensors']

def init_logger() -> logging.Logger:
    if False:
        while True:
            i = 10
    logger = logging.getLogger(__name__)
    level = logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s')
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger
logger = init_logger()

def dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    if False:
        print('Hello World!')
    all_plans = list(all_plans)
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    for (plan_idx, plan) in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)
    replicated_items = {k: v for (k, v) in key_to_plan.items() if len(v) > 1}
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    for (key, plans) in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)
    logger.info('Duplicate keys to remove: %s', plan_to_keys)
    for (plan_idx, keys) in plan_to_keys.items():
        key_set = set(keys)
        new_items = [write_item for write_item in all_plans[plan_idx].items if write_item.index not in key_set]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)
    return all_plans