"""Module for dataclasses to hold metadata of cacheable PCollections in the user
code scope.

For internal use only; no backwards-compatibility guarantees.
"""
from dataclasses import dataclass
import apache_beam as beam

@dataclass
class Cacheable:
    var: str
    version: str
    producer_version: str
    pcoll: beam.pvalue.PCollection

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.var, self.version, self.producer_version, self.pcoll))

    @staticmethod
    def from_pcoll(pcoll_name: str, pcoll: beam.pvalue.PCollection) -> 'Cacheable':
        if False:
            while True:
                i = 10
        return Cacheable(pcoll_name, str(id(pcoll)), str(id(pcoll.producer)), pcoll)

    def to_key(self):
        if False:
            return 10
        return CacheKey(self.var, self.version, self.producer_version, str(id(self.pcoll.pipeline)))

@dataclass
class CacheKey:
    """The identifier of a cacheable PCollection in cache.

  It contains 4 stringified components:
  var: The obfuscated variable name of the PCollection.
  version: The id() of the PCollection.
  producer_version: The id() of the producer of the PCollection.
  pipeline_id: The id() of the pipeline the PCollection belongs to.
  """
    var: str
    version: str
    producer_version: str
    pipeline_id: str

    def __post_init__(self):
        if False:
            while True:
                i = 10
        from apache_beam.runners.interactive.utils import obfuscate
        self.var = obfuscate(self.var)[:10]

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.var, self.version, self.producer_version, self.pipeline_id))

    @staticmethod
    def from_str(r: str) -> 'CacheKey':
        if False:
            i = 10
            return i + 15
        r_split = r.split('-')
        ck = CacheKey(*r_split)
        ck.var = r_split[0]
        return ck

    @staticmethod
    def from_pcoll(pcoll_name: str, pcoll: beam.pvalue.PCollection) -> 'CacheKey':
        if False:
            return 10
        return CacheKey(pcoll_name, str(id(pcoll)), str(id(pcoll.producer)), str(id(pcoll.pipeline)))

    def to_str(self):
        if False:
            i = 10
            return i + 15
        return '-'.join([self.var, self.version, self.producer_version, self.pipeline_id])

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.to_str()

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.to_str()