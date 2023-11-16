from dataclasses import dataclass, field
from enum import Enum
from typing import List
from mage_ai.services.spark.models.applications import Application
from mage_ai.services.spark.models.base import BaseSparkModel
from mage_ai.services.spark.models.jobs import Job
from mage_ai.services.spark.models.stages import StageAttempt

class SqlStatus(str, Enum):
    COMPLETED = 'COMPLETED'

@dataclass
class Metric(BaseSparkModel):
    name: str = None
    value: str = None

@dataclass
class Edge(BaseSparkModel):
    from_id: int = None
    to_id: int = None

@dataclass
class Node(BaseSparkModel):
    metrics: List[Metric] = field(default_factory=list)
    node_id: int = None
    node_name: str = None
    whole_stage_codegen_id: int = None

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.metrics:
            self.metrics = [Metric.load(**model) for model in self.metrics]

    @property
    def id(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.node_id

@dataclass
class Sql(BaseSparkModel):
    application: Application = None
    description: str = None
    duration: int = None
    edges: List[Edge] = field(default_factory=list)
    failed_job_ids: List[int] = field(default_factory=list)
    id: int = None
    jobs: List[Job] = field(default_factory=list)
    '\n    == Physical Plan ==\n    AdaptiveSparkPlan (27)\n    +- == Final Plan ==\n       * HashAggregate (18)\n       +- * HashAggregate (17)\n          +- * Project (16)\n             +- * BroadcastHashJoin Inner BuildLeft (15)\n                :- BroadcastQueryStage (8), Statistics(sizeInBytes=1035.2 KiB, rowCount=1.00E+4)\n                :  +- BroadcastExchange (7)\n                :     +- AQEShuffleRead (6)\n                :        +- ShuffleQueryStage (5), Statistics(sizeInBytes=625.0 KiB, rowCount=1.00\n                :           +- Exchange (4)\n                :              +- * Project (3)\n                :                 +- * Filter (2)\n                :                    +- * Scan ExistingRDD (1)\n                +- AQEShuffleRead (14)\n                   +- ShuffleQueryStage (13), Statistics(sizeInBytes=156.3 KiB, rowCount=1.00E+4)\n                      +- Exchange (12)\n                         +- * Project (11)\n                            +- * Filter (10)\n                               +- * Scan ExistingRDD (9)\n    +- == Initial Plan ==\n       HashAggregate (26)\n       +- HashAggregate (25)\n          +- Project (24)\n             +- SortMergeJoin Inner (23)\n                :- Sort (20)\n                :  +- Exchange (19)\n                :     +- Project (3)\n                :        +- Filter (2)\n                :           +- Scan ExistingRDD (1)\n                +- Sort (22)\n                   +- Exchange (21)\n                      +- Project (11)\n                         +- Filter (10)\n                            +- Scan ExistingRDD (9)\n    '
    nodes: List[Node] = field(default_factory=list)
    plan_description: str = None
    running_job_ids: List[int] = field(default_factory=list)
    stages: List[StageAttempt] = field(default_factory=list)
    status: int = SqlStatus
    submission_time: str = None
    success_job_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        if self.application and isinstance(self.application, dict):
            self.application = Application.load(**self.application)
        if self.edges:
            self.edges = [Edge.load(**edge) for edge in self.edges]
        if self.jobs:
            self.jobs = [Job.load(**m) for m in self.jobs]
        if self.nodes:
            self.nodes = [Node.load(**node) for node in self.nodes]
        if self.stages:
            self.stages = [StageAttempt.load(**m) for m in self.stages]
        if self.status:
            try:
                self.status = SqlStatus(self.status)
            except ValueError as err:
                print(f'[WARNING] Thread: {err}')
                self.status = self.status