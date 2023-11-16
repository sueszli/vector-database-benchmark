"""Class that tracks derived/pipeline fragments from user pipelines.

For internal use only; no backwards-compatibility guarantees.
In the InteractiveRunner the design is to keep the user pipeline unchanged,
create a copy of the user pipeline, and modify the copy. When the derived
pipeline runs, there should only be per-user pipeline state. This makes sure
that derived pipelines can link back to the parent user pipeline.
"""
import shutil
from typing import Iterator
from typing import Optional
import apache_beam as beam

class UserPipelineTracker:
    """Tracks user pipelines from derived pipelines.

  This data structure is similar to a disjoint set data structure. A derived
  pipeline can only have one parent user pipeline. A user pipeline can have many
  derived pipelines.
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._user_pipelines: dict[beam.Pipeline, list[beam.Pipeline]] = {}
        self._derived_pipelines: dict[beam.Pipeline] = {}
        self._pid_to_pipelines: dict[beam.Pipeline] = {}

    def __iter__(self) -> Iterator[beam.Pipeline]:
        if False:
            i = 10
            return i + 15
        'Iterates through all the user pipelines.'
        for p in self._user_pipelines:
            yield p

    def _key(self, pipeline: beam.Pipeline) -> str:
        if False:
            for i in range(10):
                print('nop')
        return str(id(pipeline))

    def evict(self, pipeline: beam.Pipeline) -> None:
        if False:
            return 10
        'Evicts the pipeline.\n\n    Removes the given pipeline and derived pipelines if a user pipeline.\n    Otherwise, removes the given derived pipeline.\n    '
        user_pipeline = self.get_user_pipeline(pipeline)
        if user_pipeline:
            for d in self._user_pipelines[user_pipeline]:
                del self._derived_pipelines[d]
            del self._user_pipelines[user_pipeline]
        elif pipeline in self._derived_pipelines:
            del self._derived_pipelines[pipeline]

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        'Clears the tracker of all user and derived pipelines.'
        for p in self._pid_to_pipelines.values():
            shutil.rmtree(p.local_tempdir, ignore_errors=True)
        self._user_pipelines.clear()
        self._derived_pipelines.clear()
        self._pid_to_pipelines.clear()

    def get_pipeline(self, pid: str) -> Optional[beam.Pipeline]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the pipeline corresponding to the given pipeline id.'
        return self._pid_to_pipelines.get(pid, None)

    def add_user_pipeline(self, p: beam.Pipeline) -> beam.Pipeline:
        if False:
            while True:
                i = 10
        'Adds a user pipeline with an empty set of derived pipelines.'
        self._memoize_pipieline(p)
        user_pipeline = self.get_user_pipeline(p)
        if not user_pipeline:
            user_pipeline = p
            self._user_pipelines[p] = []
        return user_pipeline

    def _memoize_pipieline(self, p: beam.Pipeline) -> None:
        if False:
            while True:
                i = 10
        'Memoizes the pid of the pipeline to the pipeline object.'
        pid = self._key(p)
        if pid not in self._pid_to_pipelines:
            self._pid_to_pipelines[pid] = p

    def add_derived_pipeline(self, maybe_user_pipeline: beam.Pipeline, derived_pipeline: beam.Pipeline) -> None:
        if False:
            i = 10
            return i + 15
        'Adds a derived pipeline with the user pipeline.\n\n    If the `maybe_user_pipeline` is a user pipeline, then the derived pipeline\n    will be added to its set. Otherwise, the derived pipeline will be added to\n    the user pipeline of the `maybe_user_pipeline`.\n\n    By doing the above one can do:\n    p = beam.Pipeline()\n\n    derived1 = beam.Pipeline()\n    derived2 = beam.Pipeline()\n\n    ut = UserPipelineTracker()\n    ut.add_derived_pipeline(p, derived1)\n    ut.add_derived_pipeline(derived1, derived2)\n\n    # Returns p.\n    ut.get_user_pipeline(derived2)\n    '
        self._memoize_pipieline(maybe_user_pipeline)
        self._memoize_pipieline(derived_pipeline)
        assert derived_pipeline not in self._derived_pipelines
        user = self.add_user_pipeline(maybe_user_pipeline)
        self._derived_pipelines[derived_pipeline] = user
        self._user_pipelines[user].append(derived_pipeline)

    def get_user_pipeline(self, p: beam.Pipeline) -> Optional[beam.Pipeline]:
        if False:
            i = 10
            return i + 15
        'Returns the user pipeline of the given pipeline.\n\n    If the given pipeline has no user pipeline, i.e. not added to this tracker,\n    then this returns None. If the given pipeline is a user pipeline then this\n    returns the same pipeline. If the given pipeline is a derived pipeline then\n    this returns the user pipeline.\n    '
        if p in self._user_pipelines:
            return p
        if p in self._derived_pipelines:
            return self._derived_pipelines[p]
        return None