"""Base classes for all of Oppia's Apache Beam jobs.

Jobs are composed of the following components:
    - Pipelines
    - PTransforms
    - PValues/PCollections
    - Runners

Pipelines manage a DAG (directed acyclic graph) of PValues and the PTransforms
that compute them. Conceptually, PValues are the DAG's nodes and PTransforms are
the edges.

For example:

    .------------. io.ReadFromText(fname) .-------. FlatMap(str.split)
    | Input File | ---------------------> | Lines | -----------------.
    '------------'                        '-------'                  |
                                                                     |
       .----------------. combiners.Count.PerElement() .-------.     |
    .- | (word, count)s | <--------------------------- | Words | <---'
    |  '----------------'                              '-------'
    |
    | MapTuple(lambda word, count: '%s: %d' % (word, count)) .------------.
    '------------------------------------------------------> | "word: #"s |
                                                             '------------'
                                                                    |
                             .-------------. io.WriteToText(ofname) |
                             | Output File | <----------------------'
                             '-------------'

PCollections are PValues that represent a dataset of (virtually) any size,
including unbounded/continuous datasets. They are the primary input and output
types used by PTransforms.

Runners provide a run() method for visiting every node (PValue) in the
pipeline's DAG by executing the edges (PTransforms) to compute their values. At
Oppia, we use DataflowRunner() to have our Pipelines run on the Google Cloud
Dataflow service: https://cloud.google.com/dataflow.
"""
from __future__ import annotations
from core.jobs.types import job_run_result
import apache_beam as beam
from typing import Dict, List, Tuple, Type, cast

class JobMetaclass(type):
    """Metaclass for all of Oppia's Apache Beam jobs.

    This class keeps track of the complete list of jobs. The list can be read
    with the JobMetaclass.get_all_jobs() class method.

    THIS CLASS IS AN IMPLEMENTATION DETAIL, DO NOT USE IT DIRECTLY. All user
    code should simply inherit from the JobBase class, found below.
    """
    _JOB_REGISTRY: Dict[str, Type[JobBase]] = {}

    def __new__(mcs: Type[JobMetaclass], name: str, bases: Tuple[type, ...], namespace: Dict[str, str]) -> JobMetaclass:
        if False:
            while True:
                i = 10
        'Creates a new job class with type `JobMetaclass`.\n\n        https://docs.python.org/3/reference/datamodel.html#customizing-class-creation\n\n        This metaclass adds jobs to the _JOB_REGISTRY dict, keyed by name, as\n        they are created. We use the registry to reject jobs with duplicate\n        names and to provide the convenient: JobMetaclass.get_all_jobs().\n\n        We use a metaclass instead of other alternatives (like decorators or a\n        manual list), because metaclasses cannot be forgotten to be used,\n        whereas the other alternatives can, and because they do not need help\n        from third party linters to be enforced.\n\n        Args:\n            name: str. The name of the class.\n            bases: tuple(type). The sequence of base classes for the new class.\n            namespace: dict(str: *). The namespace of the class. This is where\n                the methods, functions, and attributes of the class are kept.\n\n        Returns:\n            class. The new class instance.\n\n        Raises:\n            TypeError. The given name is already in use.\n            TypeError. The given name must end with "Job".\n            TypeError. The class with the given name must inherit from JobBase.\n        '
        if name in mcs._JOB_REGISTRY:
            collision = mcs._JOB_REGISTRY[name]
            raise TypeError('%s name is already used by %s.%s' % (name, collision.__module__, name))
        job_cls = super(JobMetaclass, mcs).__new__(mcs, name, bases, namespace)
        if name == 'JobBase':
            return cast(JobMetaclass, job_cls)
        if not name.endswith('Base'):
            if issubclass(job_cls, JobBase):
                if not name.endswith('Job'):
                    raise TypeError('Job name "%s" must end with "Job"' % name)
                mcs._JOB_REGISTRY[name] = job_cls
            else:
                raise TypeError('%s must inherit from JobBase' % name)
        return cast(JobMetaclass, job_cls)

    @classmethod
    def get_all_jobs(mcs) -> List[Type[JobBase]]:
        if False:
            while True:
                i = 10
        'Returns all jobs that have inherited from the JobBase class.\n\n        Returns:\n            list(class). The classes that have inherited from JobBase.\n        '
        return list(mcs._JOB_REGISTRY.values())

    @classmethod
    def get_all_job_names(mcs) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the names of all jobs that have inherited from the JobBase\n        class.\n\n        Returns:\n            list(str). The names of all classes that hae inherited from JobBase.\n        '
        return list(mcs._JOB_REGISTRY.keys())

    @classmethod
    def get_job_class_by_name(mcs, job_name: str) -> Type[JobBase]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the class associated with the given job name.\n\n        Args:\n            job_name: str. The name of the job to return.\n\n        Returns:\n            class. The class associated to the given job name.\n\n        Raises:\n            ValueError. Given job name is not registered as a job.\n        '
        if job_name not in mcs._JOB_REGISTRY:
            raise ValueError('%s is not registered as a job' % job_name)
        return mcs._JOB_REGISTRY[job_name]

class JobBase(metaclass=JobMetaclass):
    """The base class for all of Oppia's Apache Beam jobs.

    Example:
        class CountAllModelsJob(JobBase):
            def run(self):
                return (
                    self.pipeline
                    | self.job_options.model_getter()
                    | beam.GroupBy(job_utils.get_model_kind)
                    | beam.combiners.Count.PerElement()
                )

        options = job_options.JobOptions(model_getter=model_io.GetModels)
        with pipeline.Pipeline(options=options) as p:
            count_all_models_job = CountAllModelsJob(p)
            beam_testing_util.assert_that(
                count_all_models_job.run(),
                beam_testing_util.equal_to([
                    ('BaseModel', 1),
                ]))
    """

    def __init__(self, pipeline: beam.Pipeline) -> None:
        if False:
            while True:
                i = 10
        'Initializes a new job.\n\n        Args:\n            pipeline: beam.Pipeline. The pipeline that manages the job.\n        '
        self.pipeline = pipeline

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            i = 10
            return i + 15
        'Runs PTransforms with self.pipeline to compute/process PValues.\n\n        Raises:\n            NotImplementedError. Needs to be overridden by a subclass.\n        '
        raise NotImplementedError('Subclasses must implement the run() method')