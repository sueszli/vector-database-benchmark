"""Sources and sinks.

A Source manages record-oriented data input from a particular kind of source
(e.g. a set of files, a database table, etc.). The reader() method of a source
returns a reader object supporting the iterator protocol; iteration yields
raw records of unprocessed, serialized data.


A Sink manages record-oriented data output to a particular kind of sink
(e.g. a set of files, a database table, etc.). The writer() method of a sink
returns a writer object supporting writing records of serialized data to
the sink.
"""
import logging
import math
import random
import uuid
from collections import namedtuple
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Union
from apache_beam import coders
from apache_beam import pvalue
from apache_beam.coders.coders import _MemoizingPickleCoder
from apache_beam.internal import pickler
from apache_beam.portability import common_urns
from apache_beam.portability import python_urns
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.pvalue import AsIter
from apache_beam.pvalue import AsSingleton
from apache_beam.transforms import Impulse
from apache_beam.transforms import PTransform
from apache_beam.transforms import core
from apache_beam.transforms import ptransform
from apache_beam.transforms import window
from apache_beam.transforms.display import DisplayDataItem
from apache_beam.transforms.display import HasDisplayData
from apache_beam.utils import timestamp
from apache_beam.utils import urns
from apache_beam.utils.windowed_value import WindowedValue
__all__ = ['BoundedSource', 'RangeTracker', 'Read', 'RestrictionProgress', 'RestrictionTracker', 'WatermarkEstimator', 'Sink', 'Write', 'Writer']
_LOGGER = logging.getLogger(__name__)
SourceBundle = namedtuple('SourceBundle', 'weight source start_position stop_position')

class SourceBase(HasDisplayData, urns.RunnerApiFn):
    """Base class for all sources that can be passed to beam.io.Read(...).
  """
    urns.RunnerApiFn.register_pickle_urn(python_urns.PICKLED_SOURCE)

    def default_output_coder(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def is_bounded(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class BoundedSource(SourceBase):
    """A source that reads a finite amount of input records.

  This class defines following operations which can be used to read the source
  efficiently.

  * Size estimation - method ``estimate_size()`` may return an accurate
    estimation in bytes for the size of the source.
  * Splitting into bundles of a given size - method ``split()`` can be used to
    split the source into a set of sub-sources (bundles) based on a desired
    bundle size.
  * Getting a RangeTracker - method ``get_range_tracker()`` should return a
    ``RangeTracker`` object for a given position range for the position type
    of the records returned by the source.
  * Reading the data - method ``read()`` can be used to read data from the
    source while respecting the boundaries defined by a given
    ``RangeTracker``.

  A runner will perform reading the source in two steps.

  (1) Method ``get_range_tracker()`` will be invoked with start and end
      positions to obtain a ``RangeTracker`` for the range of positions the
      runner intends to read. Source must define a default initial start and end
      position range. These positions must be used if the start and/or end
      positions passed to the method ``get_range_tracker()`` are ``None``
  (2) Method read() will be invoked with the ``RangeTracker`` obtained in the
      previous step.

  **Mutability**

  A ``BoundedSource`` object should not be mutated while
  its methods (for example, ``read()``) are being invoked by a runner. Runner
  implementations may invoke methods of ``BoundedSource`` objects through
  multi-threaded and/or reentrant execution modes.
  """

    def estimate_size(self):
        if False:
            i = 10
            return i + 15
        'Estimates the size of source in bytes.\n\n    An estimate of the total size (in bytes) of the data that would be read\n    from this source. This estimate is in terms of external storage size,\n    before performing decompression or other processing.\n\n    Returns:\n      estimated size of the source if the size can be determined, ``None``\n      otherwise.\n    '
        raise NotImplementedError

    def split(self, desired_bundle_size, start_position=None, stop_position=None):
        if False:
            return 10
        "Splits the source into a set of bundles.\n\n    Bundles should be approximately of size ``desired_bundle_size`` bytes.\n\n    Args:\n      desired_bundle_size: the desired size (in bytes) of the bundles returned.\n      start_position: if specified the given position must be used as the\n                      starting position of the first bundle.\n      stop_position: if specified the given position must be used as the ending\n                     position of the last bundle.\n    Returns:\n      an iterator of objects of type 'SourceBundle' that gives information about\n      the generated bundles.\n    "
        raise NotImplementedError

    def get_range_tracker(self, start_position, stop_position):
        if False:
            print('Hello World!')
        "Returns a RangeTracker for a given position range.\n\n    Framework may invoke ``read()`` method with the RangeTracker object returned\n    here to read data from the source.\n\n    Args:\n      start_position: starting position of the range. If 'None' default start\n                      position of the source must be used.\n      stop_position:  ending position of the range. If 'None' default stop\n                      position of the source must be used.\n    Returns:\n      a ``RangeTracker`` for the given position range.\n    "
        raise NotImplementedError

    def read(self, range_tracker):
        if False:
            i = 10
            return i + 15
        'Returns an iterator that reads data from the source.\n\n    The returned set of data must respect the boundaries defined by the given\n    ``RangeTracker`` object. For example:\n\n      * Returned set of data must be for the range\n        ``[range_tracker.start_position, range_tracker.stop_position)``. Note\n        that a source may decide to return records that start after\n        ``range_tracker.stop_position``. See documentation in class\n        ``RangeTracker`` for more details. Also, note that framework might\n        invoke ``range_tracker.try_split()`` to perform dynamic split\n        operations. range_tracker.stop_position may be updated\n        dynamically due to successful dynamic split operations.\n      * Method ``range_tracker.try_split()`` must be invoked for every record\n        that starts at a split point.\n      * Method ``range_tracker.record_current_position()`` may be invoked for\n        records that do not start at split points.\n\n    Args:\n      range_tracker: a ``RangeTracker`` whose boundaries must be respected\n                     when reading data from the source. A runner that reads this\n                     source muss pass a ``RangeTracker`` object that is not\n                     ``None``.\n    Returns:\n      an iterator of data read by the source.\n    '
        raise NotImplementedError

    def default_output_coder(self):
        if False:
            return 10
        'Coder that should be used for the records returned by the source.\n\n    Should be overridden by sources that produce objects that can be encoded\n    more efficiently than pickling.\n    '
        return coders.registry.get_coder(object)

    def is_bounded(self):
        if False:
            return 10
        return True

class RangeTracker(object):
    """A thread safe object used by Dataflow source framework.

  A Dataflow source is defined using a ''BoundedSource'' and a ''RangeTracker''
  pair. A ''RangeTracker'' is used by Dataflow source framework to perform
  dynamic work rebalancing of position-based sources.

  **Position-based sources**

  A position-based source is one where the source can be described by a range
  of positions of an ordered type and the records returned by the reader can be
  described by positions of the same type.

  In case a record occupies a range of positions in the source, the most
  important thing about the record is the position where it starts.

  Defining the semantics of positions for a source is entirely up to the source
  class, however the chosen definitions have to obey certain properties in order
  to make it possible to correctly split the source into parts, including
  dynamic splitting. Two main aspects need to be defined:

  1. How to assign starting positions to records.
  2. Which records should be read by a source with a range '[A, B)'.

  Moreover, reading a range must be *efficient*, i.e., the performance of
  reading a range should not significantly depend on the location of the range.
  For example, reading the range [A, B) should not require reading all data
  before 'A'.

  The sections below explain exactly what properties these definitions must
  satisfy, and how to use a ``RangeTracker`` with a properly defined source.

  **Properties of position-based sources**

  The main requirement for position-based sources is *associativity*: reading
  records from '[A, B)' and records from '[B, C)' should give the same
  records as reading from '[A, C)', where 'A <= B <= C'. This property
  ensures that no matter how a range of positions is split into arbitrarily many
  sub-ranges, the total set of records described by them stays the same.

  The other important property is how the source's range relates to positions of
  records in the source. In many sources each record can be identified by a
  unique starting position. In this case:

  * All records returned by a source '[A, B)' must have starting positions in
    this range.
  * All but the last record should end within this range. The last record may or
    may not extend past the end of the range.
  * Records should not overlap.

  Such sources should define "read '[A, B)'" as "read from the first record
  starting at or after 'A', up to but not including the first record starting
  at or after 'B'".

  Some examples of such sources include reading lines or CSV from a text file,
  reading keys and values from a BigTable, etc.

  The concept of *split points* allows to extend the definitions for dealing
  with sources where some records cannot be identified by a unique starting
  position.

  In all cases, all records returned by a source '[A, B)' must *start* at or
  after 'A'.

  **Split points**

  Some sources may have records that are not directly addressable. For example,
  imagine a file format consisting of a sequence of compressed blocks. Each
  block can be assigned an offset, but records within the block cannot be
  directly addressed without decompressing the block. Let us refer to this
  hypothetical format as <i>CBF (Compressed Blocks Format)</i>.

  Many such formats can still satisfy the associativity property. For example,
  in CBF, reading '[A, B)' can mean "read all the records in all blocks whose
  starting offset is in '[A, B)'".

  To support such complex formats, we introduce the notion of *split points*. We
  say that a record is a split point if there exists a position 'A' such that
  the record is the first one to be returned when reading the range
  '[A, infinity)'. In CBF, the only split points would be the first records
  in each block.

  Split points allow us to define the meaning of a record's position and a
  source's range in all cases:

  * For a record that is at a split point, its position is defined to be the
    largest 'A' such that reading a source with the range '[A, infinity)'
    returns this record.
  * Positions of other records are only required to be non-decreasing.
  * Reading the source '[A, B)' must return records starting from the first
    split point at or after 'A', up to but not including the first split point
    at or after 'B'. In particular, this means that the first record returned
    by a source MUST always be a split point.
  * Positions of split points must be unique.

  As a result, for any decomposition of the full range of the source into
  position ranges, the total set of records will be the full set of records in
  the source, and each record will be read exactly once.

  **Consumed positions**

  As the source is being read, and records read from it are being passed to the
  downstream transforms in the pipeline, we say that positions in the source are
  being *consumed*. When a reader has read a record (or promised to a caller
  that a record will be returned), positions up to and including the record's
  start position are considered *consumed*.

  Dynamic splitting can happen only at *unconsumed* positions. If the reader
  just returned a record at offset 42 in a file, dynamic splitting can happen
  only at offset 43 or beyond, as otherwise that record could be read twice (by
  the current reader and by a reader of the task starting at 43).
  """
    SPLIT_POINTS_UNKNOWN = object()

    def start_position(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the starting position of the current range, inclusive.'
        raise NotImplementedError(type(self))

    def stop_position(self):
        if False:
            return 10
        'Returns the ending position of the current range, exclusive.'
        raise NotImplementedError(type(self))

    def try_claim(self, position):
        if False:
            i = 10
            return i + 15
        'Atomically determines if a record at a split point is within the range.\n\n    This method should be called **if and only if** the record is at a split\n    point. This method may modify the internal state of the ``RangeTracker`` by\n    updating the last-consumed position to ``position``.\n\n    ** Thread safety **\n\n    Methods of the class ``RangeTracker`` including this method may get invoked\n    by different threads, hence must be made thread-safe, e.g. by using a single\n    lock object.\n\n    Args:\n      position: starting position of a record being read by a source.\n\n    Returns:\n      ``True``, if the given position falls within the current range, returns\n      ``False`` otherwise.\n    '
        raise NotImplementedError

    def set_current_position(self, position):
        if False:
            print('Hello World!')
        'Updates the last-consumed position to the given position.\n\n    A source may invoke this method for records that do not start at split\n    points. This may modify the internal state of the ``RangeTracker``. If the\n    record starts at a split point, method ``try_claim()`` **must** be invoked\n    instead of this method.\n\n    Args:\n      position: starting position of a record being read by a source.\n    '
        raise NotImplementedError

    def position_at_fraction(self, fraction):
        if False:
            for i in range(10):
                print('nop')
        'Returns the position at the given fraction.\n\n    Given a fraction within the range [0.0, 1.0) this method will return the\n    position at the given fraction compared to the position range\n    [self.start_position, self.stop_position).\n\n    ** Thread safety **\n\n    Methods of the class ``RangeTracker`` including this method may get invoked\n    by different threads, hence must be made thread-safe, e.g. by using a single\n    lock object.\n\n    Args:\n      fraction: a float value within the range [0.0, 1.0).\n    Returns:\n      a position within the range [self.start_position, self.stop_position).\n    '
        raise NotImplementedError

    def try_split(self, position):
        if False:
            print('Hello World!')
        'Atomically splits the current range.\n\n    Determines a position to split the current range, split_position, based on\n    the given position. In most cases split_position and position will be the\n    same.\n\n    Splits the current range \'[self.start_position, self.stop_position)\'\n    into a "primary" part \'[self.start_position, split_position)\' and a\n    "residual" part \'[split_position, self.stop_position)\', assuming the\n    current last-consumed position is within\n    \'[self.start_position, split_position)\' (i.e., split_position has not been\n    consumed yet).\n\n    If successful, updates the current range to be the primary and returns a\n    tuple (split_position, split_fraction). split_fraction should be the\n    fraction of size of range \'[self.start_position, split_position)\' compared\n    to the original (before split) range\n    \'[self.start_position, self.stop_position)\'.\n\n    If the split_position has already been consumed, returns ``None``.\n\n    ** Thread safety **\n\n    Methods of the class ``RangeTracker`` including this method may get invoked\n    by different threads, hence must be made thread-safe, e.g. by using a single\n    lock object.\n\n    Args:\n      position: suggested position where the current range should try to\n                be split at.\n    Returns:\n      a tuple containing the split position and split fraction if split is\n      successful. Returns ``None`` otherwise.\n    '
        raise NotImplementedError

    def fraction_consumed(self):
        if False:
            while True:
                i = 10
        "Returns the approximate fraction of consumed positions in the source.\n\n    ** Thread safety **\n\n    Methods of the class ``RangeTracker`` including this method may get invoked\n    by different threads, hence must be made thread-safe, e.g. by using a single\n    lock object.\n\n    Returns:\n      the approximate fraction of positions that have been consumed by\n      successful 'try_split()' and  'try_claim()'  calls, or\n      0.0 if no such calls have happened.\n    "
        raise NotImplementedError

    def split_points(self):
        if False:
            i = 10
            return i + 15
        'Gives the number of split points consumed and remaining.\n\n    For a ``RangeTracker`` used by a ``BoundedSource`` (within a\n    ``BoundedSource.read()`` invocation) this method produces a 2-tuple that\n    gives the number of split points consumed by the ``BoundedSource`` and the\n    number of split points remaining within the range of the ``RangeTracker``\n    that has not been consumed by the ``BoundedSource``.\n\n    More specifically, given that the position of the current record being read\n    by ``BoundedSource`` is current_position this method produces a tuple that\n    consists of\n    (1) number of split points in the range [self.start_position(),\n    current_position) without including the split point that is currently being\n    consumed. This represents the total amount of parallelism in the consumed\n    part of the source.\n    (2) number of split points within the range\n    [current_position, self.stop_position()) including the split point that is\n    currently being consumed. This represents the total amount of parallelism in\n    the unconsumed part of the source.\n\n    Methods of the class ``RangeTracker`` including this method may get invoked\n    by different threads, hence must be made thread-safe, e.g. by using a single\n    lock object.\n\n    ** General information about consumed and remaining number of split\n       points returned by this method. **\n\n      * Before a source read (``BoundedSource.read()`` invocation) claims the\n        first split point, number of consumed split points is 0. This condition\n        holds independent of whether the input is "splittable". A splittable\n        source is a source that has more than one split point.\n      * Any source read that has only claimed one split point has 0 consumed\n        split points since the first split point is the current split point and\n        is still being processed. This condition holds independent of whether\n        the input is splittable.\n      * For an empty source read which never invokes\n        ``RangeTracker.try_claim()``, the consumed number of split points is 0.\n        This condition holds independent of whether the input is splittable.\n      * For a source read which has invoked ``RangeTracker.try_claim()`` n\n        times, the consumed number of split points is  n -1.\n      * If a ``BoundedSource`` sets a callback through function\n        ``set_split_points_unclaimed_callback()``, ``RangeTracker`` can use that\n        callback when determining remaining number of split points.\n      * Remaining split points should include the split point that is currently\n        being consumed by the source read. Hence if the above callback returns\n        an integer value n, remaining number of split points should be (n + 1).\n      * After last split point is claimed remaining split points becomes 1,\n        because this unfinished read itself represents an  unfinished split\n        point.\n      * After all records of the source has been consumed, remaining number of\n        split points becomes 0 and consumed number of split points becomes equal\n        to the total number of split points within the range being read by the\n        source. This method does not address this condition and will continue to\n        report number of consumed split points as\n        ("total number of split points" - 1) and number of remaining split\n        points as 1. A runner that performs the reading of the source can\n        detect when all records have been consumed and adjust remaining and\n        consumed number of split points accordingly.\n\n    ** Examples **\n\n    (1) A "perfectly splittable" input which can be read in parallel down to the\n        individual records.\n\n        Consider a perfectly splittable input that consists of 50 split points.\n\n      * Before a source read (``BoundedSource.read()`` invocation) claims the\n        first split point, number of consumed split points is 0 number of\n        remaining split points is 50.\n      * After claiming first split point, consumed number of split points is 0\n        and remaining number of split is 50.\n      * After claiming split point #30, consumed number of split points is 29\n        and remaining number of split points is 21.\n      * After claiming all 50 split points, consumed number of split points is\n        49 and remaining number of split points is 1.\n\n    (2) a "block-compressed" file format such as ``avroio``, in which a block of\n        records has to be read as a whole, but different blocks can be read in\n        parallel.\n\n        Consider a block compressed input that consists of 5 blocks.\n\n      * Before a source read (``BoundedSource.read()`` invocation) claims the\n        first split point (first block), number of consumed split points is 0\n        number of remaining split points is 5.\n      * After claiming first split point, consumed number of split points is 0\n        and remaining number of split is 5.\n      * After claiming split point #3, consumed number of split points is 2\n        and remaining number of split points is 3.\n      * After claiming all 5 split points, consumed number of split points is\n        4 and remaining number of split points is 1.\n\n    (3) an "unsplittable" input such as a cursor in a database or a gzip\n        compressed file.\n\n        Such an input is considered to have only a single split point. Number of\n        consumed split points is always 0 and number of remaining split points\n        is always 1.\n\n    By default ``RangeTracker` returns ``RangeTracker.SPLIT_POINTS_UNKNOWN`` for\n    both consumed and remaining number of split points, which indicates that the\n    number of split points consumed and remaining is unknown.\n\n    Returns:\n      A pair that gives consumed and remaining number of split points. Consumed\n      number of split points should be an integer larger than or equal to zero\n      or ``RangeTracker.SPLIT_POINTS_UNKNOWN``. Remaining number of split points\n      should be an integer larger than zero or\n      ``RangeTracker.SPLIT_POINTS_UNKNOWN``.\n    '
        return (RangeTracker.SPLIT_POINTS_UNKNOWN, RangeTracker.SPLIT_POINTS_UNKNOWN)

    def set_split_points_unclaimed_callback(self, callback):
        if False:
            i = 10
            return i + 15
        'Sets a callback for determining the unclaimed number of split points.\n\n    By invoking this function, a ``BoundedSource`` can set a callback function\n    that may get invoked by the ``RangeTracker`` to determine the number of\n    unclaimed split points. A split point is unclaimed if\n    ``RangeTracker.try_claim()`` method has not been successfully invoked for\n    that particular split point. The callback function accepts a single\n    parameter, a stop position for the BoundedSource (stop_position). If the\n    record currently being consumed by the ``BoundedSource`` is at position\n    current_position, callback should return the number of split points within\n    the range (current_position, stop_position). Note that, this should not\n    include the split point that is currently being consumed by the source.\n\n    This function must be implemented by subclasses before being used.\n\n    Args:\n      callback: a function that takes a single parameter, a stop position,\n                and returns unclaimed number of split points for the source read\n                operation that is calling this function. Value returned from\n                callback should be either an integer larger than or equal to\n                zero or ``RangeTracker.SPLIT_POINTS_UNKNOWN``.\n    '
        raise NotImplementedError

class Sink(HasDisplayData):
    """This class is deprecated, no backwards-compatibility guarantees.

  A resource that can be written to using the ``beam.io.Write`` transform.

  Here ``beam`` stands for Apache Beam Python code imported in following manner.
  ``import apache_beam as beam``.

  A parallel write to an ``iobase.Sink`` consists of three phases:

  1. A sequential *initialization* phase (e.g., creating a temporary output
     directory, etc.)
  2. A parallel write phase where workers write *bundles* of records
  3. A sequential *finalization* phase (e.g., committing the writes, merging
     output files, etc.)

  Implementing a new sink requires extending two classes.

  1. iobase.Sink

  ``iobase.Sink`` is an immutable logical description of the location/resource
  to write to. Depending on the type of sink, it may contain fields such as the
  path to an output directory on a filesystem, a database table name,
  etc. ``iobase.Sink`` provides methods for performing a write operation to the
  sink described by it. To this end, implementors of an extension of
  ``iobase.Sink`` must implement three methods:
  ``initialize_write()``, ``open_writer()``, and ``finalize_write()``.

  2. iobase.Writer

  ``iobase.Writer`` is used to write a single bundle of records. An
  ``iobase.Writer`` defines two methods: ``write()`` which writes a
  single record from the bundle and ``close()`` which is called once
  at the end of writing a bundle.

  See also ``apache_beam.io.filebasedsink.FileBasedSink`` which provides a
  simpler API for writing sinks that produce files.

  **Execution of the Write transform**

  ``initialize_write()``, ``pre_finalize()``, and ``finalize_write()`` are
  conceptually called once. However, implementors must
  ensure that these methods are *idempotent*, as they may be called multiple
  times on different machines in the case of failure/retry. A method may be
  called more than once concurrently, in which case it's okay to have a
  transient failure (such as due to a race condition). This failure should not
  prevent subsequent retries from succeeding.

  ``initialize_write()`` should perform any initialization that needs to be done
  prior to writing to the sink. ``initialize_write()`` may return a result
  (let's call this ``init_result``) that contains any parameters it wants to
  pass on to its writers about the sink. For example, a sink that writes to a
  file system may return an ``init_result`` that contains a dynamically
  generated unique directory to which data should be written.

  To perform writing of a bundle of elements, Dataflow execution engine will
  create an ``iobase.Writer`` using the implementation of
  ``iobase.Sink.open_writer()``. When invoking ``open_writer()`` execution
  engine will provide the ``init_result`` returned by ``initialize_write()``
  invocation as well as a *bundle id* (let's call this ``bundle_id``) that is
  unique for each invocation of ``open_writer()``.

  Execution engine will then invoke ``iobase.Writer.write()`` implementation for
  each element that has to be written. Once all elements of a bundle are
  written, execution engine will invoke ``iobase.Writer.close()`` implementation
  which should return a result (let's call this ``write_result``) that contains
  information that encodes the result of the write and, in most cases, some
  encoding of the unique bundle id. For example, if each bundle is written to a
  unique temporary file, ``close()`` method may return an object that contains
  the temporary file name. After writing of all bundles is complete, execution
  engine will invoke ``pre_finalize()`` and then ``finalize_write()``
  implementation.

  The execution of a write transform can be illustrated using following pseudo
  code (assume that the outer for loop happens in parallel across many
  machines)::

    init_result = sink.initialize_write()
    write_results = []
    for bundle in partition(pcoll):
      writer = sink.open_writer(init_result, generate_bundle_id())
      for elem in bundle:
        writer.write(elem)
      write_results.append(writer.close())
    pre_finalize_result = sink.pre_finalize(init_result, write_results)
    sink.finalize_write(init_result, write_results, pre_finalize_result)


  **init_result**

  Methods of 'iobase.Sink' should agree on the 'init_result' type that will be
  returned when initializing the sink. This type can be a client-defined object
  or an existing type. The returned type must be picklable using Dataflow coder
  ``coders.PickleCoder``. Returning an init_result is optional.

  **bundle_id**

  In order to ensure fault-tolerance, a bundle may be executed multiple times
  (e.g., in the event of failure/retry or for redundancy). However, exactly one
  of these executions will have its result passed to the
  ``iobase.Sink.finalize_write()`` method. Each call to
  ``iobase.Sink.open_writer()`` is passed a unique bundle id when it is called
  by the ``WriteImpl`` transform, so even redundant or retried bundles will have
  a unique way of identifying their output.

  The bundle id should be used to guarantee that a bundle's output is unique.
  This uniqueness guarantee is important; if a bundle is to be output to a file,
  for example, the name of the file must be unique to avoid conflicts with other
  writers. The bundle id should be encoded in the writer result returned by the
  writer and subsequently used by the ``finalize_write()`` method to identify
  the results of successful writes.

  For example, consider the scenario where a Writer writes files containing
  serialized records and the ``finalize_write()`` is to merge or rename these
  output files. In this case, a writer may use its unique id to name its output
  file (to avoid conflicts) and return the name of the file it wrote as its
  writer result. The ``finalize_write()`` will then receive an ``Iterable`` of
  output file names that it can then merge or rename using some bundle naming
  scheme.

  **write_result**

  ``iobase.Writer.close()`` and ``finalize_write()`` implementations must agree
  on type of the ``write_result`` object returned when invoking
  ``iobase.Writer.close()``. This type can be a client-defined object or
  an existing type. The returned type must be picklable using Dataflow coder
  ``coders.PickleCoder``. Returning a ``write_result`` when
  ``iobase.Writer.close()`` is invoked is optional but if unique
  ``write_result`` objects are not returned, sink should, guarantee idempotency
  when same bundle is written multiple times due to failure/retry or redundancy.


  **More information**

  For more information on creating new sinks please refer to the official
  documentation at
  ``https://beam.apache.org/documentation/sdks/python-custom-io#creating-sinks``
  """
    skip_if_empty = False

    def initialize_write(self):
        if False:
            print('Hello World!')
        'Initializes the sink before writing begins.\n\n    Invoked before any data is written to the sink.\n\n\n    Please see documentation in ``iobase.Sink`` for an example.\n\n    Returns:\n      An object that contains any sink specific state generated by\n      initialization. This object will be passed to open_writer() and\n      finalize_write() methods.\n    '
        raise NotImplementedError

    def open_writer(self, init_result, uid):
        if False:
            for i in range(10):
                print('nop')
        'Opens a writer for writing a bundle of elements to the sink.\n\n    Args:\n      init_result: the result of initialize_write() invocation.\n      uid: a unique identifier generated by the system.\n    Returns:\n      an ``iobase.Writer`` that can be used to write a bundle of records to the\n      current sink.\n    '
        raise NotImplementedError

    def pre_finalize(self, init_result, writer_results):
        if False:
            i = 10
            return i + 15
        'Pre-finalization stage for sink.\n\n    Called after all bundle writes are complete and before finalize_write.\n    Used to setup and verify filesystem and sink states.\n\n    Args:\n      init_result: the result of ``initialize_write()`` invocation.\n      writer_results: an iterable containing results of ``Writer.close()``\n        invocations. This will only contain results of successful writes, and\n        will only contain the result of a single successful write for a given\n        bundle.\n\n    Returns:\n      An object that contains any sink specific state generated.\n      This object will be passed to finalize_write().\n    '
        raise NotImplementedError

    def finalize_write(self, init_result, writer_results, pre_finalize_result):
        if False:
            for i in range(10):
                print('nop')
        'Finalizes the sink after all data is written to it.\n\n    Given the result of initialization and an iterable of results from bundle\n    writes, performs finalization after writing and closes the sink. Called\n    after all bundle writes are complete.\n\n    The bundle write results that are passed to finalize are those returned by\n    bundles that completed successfully. Although bundles may have been run\n    multiple times (for fault-tolerance), only one writer result will be passed\n    to finalize for each bundle. An implementation of finalize should perform\n    clean up of any failed and successfully retried bundles.  Note that these\n    failed bundles will not have their writer result passed to finalize, so\n    finalize should be capable of locating any temporary/partial output written\n    by failed bundles.\n\n    If all retries of a bundle fails, the whole pipeline will fail *without*\n    finalize_write() being invoked.\n\n    A best practice is to make finalize atomic. If this is impossible given the\n    semantics of the sink, finalize should be idempotent, as it may be called\n    multiple times in the case of failure/retry or for redundancy.\n\n    Note that the iteration order of the writer results is not guaranteed to be\n    consistent if finalize is called multiple times.\n\n    Args:\n      init_result: the result of ``initialize_write()`` invocation.\n      writer_results: an iterable containing results of ``Writer.close()``\n        invocations. This will only contain results of successful writes, and\n        will only contain the result of a single successful write for a given\n        bundle.\n      pre_finalize_result: the result of ``pre_finalize()`` invocation.\n    '
        raise NotImplementedError

class Writer(object):
    """This class is deprecated, no backwards-compatibility guarantees.

  Writes a bundle of elements from a ``PCollection`` to a sink.

  A Writer  ``iobase.Writer.write()`` writes and elements to the sink while
  ``iobase.Writer.close()`` is called after all elements in the bundle have been
  written.

  See ``iobase.Sink`` for more detailed documentation about the process of
  writing to a sink.
  """

    def write(self, value):
        if False:
            i = 10
            return i + 15
        'Writes a value to the sink using the current writer.\n    '
        raise NotImplementedError

    def close(self):
        if False:
            return 10
        'Closes the current writer.\n\n    Please see documentation in ``iobase.Sink`` for an example.\n\n    Returns:\n      An object representing the writes that were performed by the current\n      writer.\n    '
        raise NotImplementedError

    def at_capacity(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns whether this writer should be considered at capacity\n    and a new one should be created.\n    '
        return False

class Read(ptransform.PTransform):
    """A transform that reads a PCollection."""
    from apache_beam.runners.pipeline_context import PipelineContext

    def __init__(self, source):
        if False:
            while True:
                i = 10
        'Initializes a Read transform.\n\n    Args:\n      source: Data source to read from.\n    '
        super().__init__()
        self.source = source

    @staticmethod
    def get_desired_chunk_size(total_size):
        if False:
            for i in range(10):
                print('nop')
        if total_size:
            chunk_size = max(1 << 20, 1000 * int(math.sqrt(total_size)))
        else:
            chunk_size = 64 << 20
        return chunk_size

    def expand(self, pbegin):
        if False:
            print('Hello World!')
        if isinstance(self.source, BoundedSource):
            coders.registry.register_coder(BoundedSource, _MemoizingPickleCoder)
            display_data = self.source.display_data() or {}
            display_data['source'] = self.source.__class__
            return pbegin | Impulse() | 'EmitSource' >> core.Map(lambda _: self.source).with_output_types(BoundedSource) | SDFBoundedSourceReader(display_data)
        elif isinstance(self.source, ptransform.PTransform):
            return pbegin.pipeline | self.source
        else:
            return pvalue.PCollection(pbegin.pipeline, is_bounded=self.source.is_bounded())

    def get_windowing(self, unused_inputs):
        if False:
            return 10
        return core.Windowing(window.GlobalWindows())

    def _infer_output_coder(self, input_type=None, input_coder=None):
        if False:
            return 10
        if isinstance(self.source, SourceBase):
            return self.source.default_output_coder()
        else:
            return None

    def display_data(self):
        if False:
            print('Hello World!')
        return {'source': DisplayDataItem(self.source.__class__, label='Read Source'), 'source_dd': self.source}

    def to_runner_api_parameter(self, context: PipelineContext) -> Tuple[str, Any]:
        if False:
            return 10
        from apache_beam.io.gcp.pubsub import _PubSubSource
        if isinstance(self.source, _PubSubSource):
            return (common_urns.composites.PUBSUB_READ.urn, beam_runner_api_pb2.PubSubReadPayload(topic=self.source.full_topic, subscription=self.source.full_subscription, timestamp_attribute=self.source.timestamp_attribute, with_attributes=self.source.with_attributes, id_attribute=self.source.id_label))
        if isinstance(self.source, BoundedSource):
            return (common_urns.deprecated_primitives.READ.urn, beam_runner_api_pb2.ReadPayload(source=self.source.to_runner_api(context), is_bounded=beam_runner_api_pb2.IsBounded.BOUNDED if self.source.is_bounded() else beam_runner_api_pb2.IsBounded.UNBOUNDED))
        elif isinstance(self.source, ptransform.PTransform):
            return self.source.to_runner_api_parameter(context)
        raise NotImplementedError('to_runner_api_parameter not implemented for type')

    @staticmethod
    def from_runner_api_parameter(transform: beam_runner_api_pb2.PTransform, payload: Union[beam_runner_api_pb2.ReadPayload, beam_runner_api_pb2.PubSubReadPayload], context: PipelineContext) -> 'Read':
        if False:
            return 10
        if transform.spec.urn == common_urns.composites.PUBSUB_READ.urn:
            assert isinstance(payload, beam_runner_api_pb2.PubSubReadPayload)
            from apache_beam.io.gcp.pubsub import _PubSubSource
            source = _PubSubSource(topic=payload.topic or None, subscription=payload.subscription or None, id_label=payload.id_attribute or None, with_attributes=payload.with_attributes, timestamp_attribute=payload.timestamp_attribute or None)
            return Read(source)
        else:
            assert isinstance(payload, beam_runner_api_pb2.ReadPayload)
            return Read(SourceBase.from_runner_api(payload.source, context))

    @staticmethod
    def _from_runner_api_parameter_read(transform: beam_runner_api_pb2.PTransform, payload: beam_runner_api_pb2.ReadPayload, context: PipelineContext) -> 'Read':
        if False:
            i = 10
            return i + 15
        'Method for type proxying when calling register_urn due to limitations\n     in type exprs in Python'
        return Read.from_runner_api_parameter(transform, payload, context)

    @staticmethod
    def _from_runner_api_parameter_pubsub_read(transform: beam_runner_api_pb2.PTransform, payload: beam_runner_api_pb2.PubSubReadPayload, context: PipelineContext) -> 'Read':
        if False:
            print('Hello World!')
        'Method for type proxying when calling register_urn due to limitations\n     in type exprs in Python'
        return Read.from_runner_api_parameter(transform, payload, context)
ptransform.PTransform.register_urn(common_urns.deprecated_primitives.READ.urn, beam_runner_api_pb2.ReadPayload, Read._from_runner_api_parameter_read)
ptransform.PTransform.register_urn(common_urns.composites.PUBSUB_READ.urn, beam_runner_api_pb2.PubSubReadPayload, Read._from_runner_api_parameter_pubsub_read)

class Write(ptransform.PTransform):
    """A ``PTransform`` that writes to a sink.

  A sink should inherit ``iobase.Sink``. Such implementations are
  handled using a composite transform that consists of three ``ParDo``s -
  (1) a ``ParDo`` performing a global initialization (2) a ``ParDo`` performing
  a parallel write and (3) a ``ParDo`` performing a global finalization. In the
  case of an empty ``PCollection``, only the global initialization and
  finalization will be performed. Currently only batch workflows support custom
  sinks.

  Example usage::

      pcollection | beam.io.Write(MySink())

  This returns a ``pvalue.PValue`` object that represents the end of the
  Pipeline.

  The sink argument may also be a full PTransform, in which case it will be
  applied directly.  This allows composite sink-like transforms (e.g. a sink
  with some pre-processing DoFns) to be used the same as all other sinks.

  This transform also supports sinks that inherit ``iobase.NativeSink``. These
  are sinks that are implemented natively by the Dataflow service and hence
  should not be updated by users. These sinks are processed using a Dataflow
  native write transform.
  """
    from apache_beam.runners.pipeline_context import PipelineContext

    def __init__(self, sink):
        if False:
            print('Hello World!')
        'Initializes a Write transform.\n\n    Args:\n      sink: Data sink to write to.\n    '
        super().__init__()
        self.sink = sink

    def display_data(self):
        if False:
            return 10
        return {'sink': self.sink.__class__, 'sink_dd': self.sink}

    def expand(self, pcoll):
        if False:
            return 10
        from apache_beam.io.gcp.pubsub import _PubSubSink
        if isinstance(self.sink, _PubSubSink):
            return pvalue.PDone(pcoll.pipeline)
        elif isinstance(self.sink, Sink):
            return pcoll | WriteImpl(self.sink)
        elif isinstance(self.sink, ptransform.PTransform):
            return pcoll | self.sink
        else:
            raise ValueError('A sink must inherit iobase.Sink, iobase.NativeSink, or be a PTransform. Received : %r' % self.sink)

    def to_runner_api_parameter(self, context: PipelineContext) -> Tuple[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        from apache_beam.io.gcp.pubsub import _PubSubSink
        if isinstance(self.sink, _PubSubSink):
            payload = beam_runner_api_pb2.PubSubWritePayload(topic=self.sink.full_topic, id_attribute=self.sink.id_label, timestamp_attribute=self.sink.timestamp_attribute)
            return (common_urns.composites.PUBSUB_WRITE.urn, payload)
        else:
            return super().to_runner_api_parameter(context)

    @staticmethod
    @ptransform.PTransform.register_urn(common_urns.composites.PUBSUB_WRITE.urn, beam_runner_api_pb2.PubSubWritePayload)
    def from_runner_api_parameter(ptransform: Any, payload: beam_runner_api_pb2.PubSubWritePayload, unused_context: PipelineContext) -> 'Write':
        if False:
            i = 10
            return i + 15
        if ptransform.spec.urn != common_urns.composites.PUBSUB_WRITE.urn:
            raise ValueError('Write transform cannot be constructed for the given proto %r', ptransform)
        if not payload.topic:
            raise NotImplementedError('from_runner_api_parameter does not handle empty or None topic')
        from apache_beam.io.gcp.pubsub import _PubSubSink
        sink = _PubSubSink(topic=payload.topic, id_label=payload.id_attribute or None, timestamp_attribute=payload.timestamp_attribute or None)
        return Write(sink)

class WriteImpl(ptransform.PTransform):
    """Implements the writing of custom sinks."""

    def __init__(self, sink):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.sink = sink

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        do_once = pcoll.pipeline | 'DoOnce' >> core.Create([None])
        init_result_coll = do_once | 'InitializeWrite' >> core.Map(lambda _, sink: sink.initialize_write(), self.sink)
        if getattr(self.sink, 'num_shards', 0):
            min_shards = self.sink.num_shards
            if min_shards == 1:
                keyed_pcoll = pcoll | core.Map(lambda x: (None, x))
            else:
                keyed_pcoll = pcoll | core.ParDo(_RoundRobinKeyFn(), count=min_shards)
            write_result_coll = keyed_pcoll | core.WindowInto(window.GlobalWindows()) | core.GroupByKey() | 'WriteBundles' >> core.ParDo(_WriteKeyedBundleDoFn(self.sink), AsSingleton(init_result_coll))
        else:
            min_shards = 1
            write_result_coll = pcoll | core.WindowInto(window.GlobalWindows()) | 'WriteBundles' >> core.ParDo(_WriteBundleDoFn(self.sink), AsSingleton(init_result_coll)) | 'Pair' >> core.Map(lambda x: (None, x)) | core.GroupByKey() | 'Extract' >> core.FlatMap(lambda x: x[1])
        pre_finalize_coll = do_once | 'PreFinalize' >> core.FlatMap(_pre_finalize, self.sink, AsSingleton(init_result_coll), AsIter(write_result_coll))
        return do_once | 'FinalizeWrite' >> core.FlatMap(_finalize_write, self.sink, AsSingleton(init_result_coll), AsIter(write_result_coll), min_shards, AsSingleton(pre_finalize_coll)).with_output_types(str)

class _WriteBundleDoFn(core.DoFn):
    """A DoFn for writing elements to an iobase.Writer.
  Opens a writer at the first element and closes the writer at finish_bundle().
  """

    def __init__(self, sink):
        if False:
            return 10
        self.sink = sink

    def display_data(self):
        if False:
            print('Hello World!')
        return {'sink_dd': self.sink}

    def start_bundle(self):
        if False:
            return 10
        self.writer = None

    def process(self, element, init_result):
        if False:
            while True:
                i = 10
        if self.writer is None:
            self.writer = self.sink.open_writer(init_result, str(uuid.uuid4()))
        self.writer.write(element)
        if self.writer.at_capacity():
            yield self.writer.close()
            self.writer = None

    def finish_bundle(self):
        if False:
            i = 10
            return i + 15
        if self.writer is not None:
            yield WindowedValue(self.writer.close(), window.GlobalWindow().max_timestamp(), [window.GlobalWindow()])

class _WriteKeyedBundleDoFn(core.DoFn):

    def __init__(self, sink):
        if False:
            print('Hello World!')
        self.sink = sink

    def display_data(self):
        if False:
            for i in range(10):
                print('nop')
        return {'sink_dd': self.sink}

    def process(self, element, init_result):
        if False:
            while True:
                i = 10
        bundle = element
        writer = self.sink.open_writer(init_result, str(uuid.uuid4()))
        for e in bundle[1]:
            writer.write(e)
        return [window.TimestampedValue(writer.close(), timestamp.MAX_TIMESTAMP)]

def _pre_finalize(unused_element, sink, init_result, write_results):
    if False:
        return 10
    return sink.pre_finalize(init_result, write_results)

def _finalize_write(unused_element, sink, init_result, write_results, min_shards, pre_finalize_results):
    if False:
        while True:
            i = 10
    write_results = list(write_results)
    extra_shards = []
    if len(write_results) < min_shards:
        if write_results or not sink.skip_if_empty:
            _LOGGER.debug('Creating %s empty shard(s).', min_shards - len(write_results))
            for _ in range(min_shards - len(write_results)):
                writer = sink.open_writer(init_result, str(uuid.uuid4()))
                extra_shards.append(writer.close())
    outputs = sink.finalize_write(init_result, write_results + extra_shards, pre_finalize_results)
    if outputs:
        return (window.TimestampedValue(v, timestamp.MAX_TIMESTAMP) for v in outputs)

class _RoundRobinKeyFn(core.DoFn):

    def start_bundle(self):
        if False:
            i = 10
            return i + 15
        self.counter = None

    def process(self, element, count):
        if False:
            while True:
                i = 10
        if self.counter is None:
            self.counter = random.randrange(0, count)
        self.counter = (1 + self.counter) % count
        yield (self.counter, element)

class RestrictionTracker(object):
    """Manages access to a restriction.

  Keeps track of the restrictions claimed part for a Splittable DoFn.

  The restriction may be modified by different threads, however the system will
  ensure sufficient locking such that no methods on the restriction tracker
  will be called concurrently.

  See following documents for more details.
  * https://s.apache.org/splittable-do-fn
  * https://s.apache.org/splittable-do-fn-python-sdk
  """

    def current_restriction(self):
        if False:
            return 10
        'Returns the current restriction.\n\n    Returns a restriction accurately describing the full range of work the\n    current ``DoFn.process()`` call will do, including already completed work.\n\n    The current restriction returned by method may be updated dynamically due\n    to due to concurrent invocation of other methods of the\n    ``RestrictionTracker``, For example, ``split()``.\n\n    This API is required to be implemented.\n\n    Returns: a restriction object.\n    '
        raise NotImplementedError

    def current_progress(self):
        if False:
            return 10
        'Returns a RestrictionProgress object representing the current progress.\n\n    This API is recommended to be implemented. The runner can do a better job\n    at parallel processing with better progress signals.\n    '
        raise NotImplementedError

    def check_done(self):
        if False:
            return 10
        'Checks whether the restriction has been fully processed.\n\n    Called by the SDK harness after iterator returned by ``DoFn.process()``\n    has been fully read.\n\n    This method must raise a `ValueError` if there is still any unclaimed work\n    remaining in the restriction when this method is invoked. Exception raised\n    must have an informative error message.\n\n    This API is required to be implemented in order to make sure no data loss\n    during SDK processing.\n\n    Returns: ``True`` if current restriction has been fully processed.\n    Raises:\n      ValueError: if there is still any unclaimed work remaining.\n    '
        raise NotImplementedError

    def try_split(self, fraction_of_remainder):
        if False:
            print('Hello World!')
        'Splits current restriction based on fraction_of_remainder.\n\n    If splitting the current restriction is possible, the current restriction is\n    split into a primary and residual restriction pair. This invocation updates\n    the ``current_restriction()`` to be the primary restriction effectively\n    having the current ``DoFn.process()`` execution responsible for performing\n    the work that the primary restriction represents. The residual restriction\n    will be executed in a separate ``DoFn.process()`` invocation (likely in a\n    different process). The work performed by executing the primary and residual\n    restrictions as separate ``DoFn.process()`` invocations MUST be equivalent\n    to the work performed as if this split never occurred.\n\n    The ``fraction_of_remainder`` should be used in a best effort manner to\n    choose a primary and residual restriction based upon the fraction of the\n    remaining work that the current ``DoFn.process()`` invocation is responsible\n    for. For example, if a ``DoFn.process()`` was reading a file with a\n    restriction representing the offset range [100, 200) and has processed up to\n    offset 130 with a fraction_of_remainder of 0.7, the primary and residual\n    restrictions returned would be [100, 179), [179, 200) (note: current_offset\n    + fraction_of_remainder * remaining_work = 130 + 0.7 * 70 = 179).\n\n    ``fraction_of_remainder`` = 0 means a checkpoint is required.\n\n    The API is recommended to be implemented for batch pipeline given that it is\n    very important for pipeline scaling and end to end pipeline execution.\n\n    The API is required to be implemented for a streaming pipeline.\n\n    Args:\n      fraction_of_remainder: A hint as to the fraction of work the primary\n        restriction should represent based upon the current known remaining\n        amount of work.\n\n    Returns:\n      (primary_restriction, residual_restriction) if a split was possible,\n      otherwise returns ``None``.\n    '
        raise NotImplementedError

    def try_claim(self, position):
        if False:
            i = 10
            return i + 15
        'Attempts to claim the block of work in the current restriction\n    identified by the given position. Each claimed position MUST be a valid\n    split point.\n\n    If this succeeds, the DoFn MUST execute the entire block of work. If it\n    fails, the ``DoFn.process()`` MUST return ``None`` without performing any\n    additional work or emitting output (note that emitting output or performing\n    work from ``DoFn.process()`` is also not allowed before the first call of\n    this method).\n\n    The API is required to be implemented.\n\n    Args:\n      position: current position that wants to be claimed.\n\n    Returns: ``True`` if the position can be claimed as current_position.\n    Otherwise, returns ``False``.\n    '
        raise NotImplementedError

    def is_bounded(self):
        if False:
            while True:
                i = 10
        'Returns whether the amount of work represented by the current restriction\n    is bounded.\n\n    The boundedness of the restriction is used to determine the default behavior\n    of how to truncate restrictions when a pipeline is being\n    `drained <https://docs.google.com/document/d/1NExwHlj-2q2WUGhSO4jTu8XGhDPmm3cllSN8IMmWci8/edit#>`_.  # pylint: disable=line-too-long\n    If the restriction is bounded, then the entire restriction will be processed\n    otherwise the restriction will be processed till a checkpoint is possible.\n\n    The API is required to be implemented.\n\n    Returns: ``True`` if the restriction represents a finite amount of work.\n    Otherwise, returns ``False``.\n    '
        raise NotImplementedError

class WatermarkEstimator(object):
    """A WatermarkEstimator which is used for estimating output_watermark based on
  the timestamp of output records or manual modifications. Please refer to
  ``watermark_estiamtors`` for commonly used watermark estimators.

  The base class provides common APIs that are called by the framework, which
  are also accessible inside a DoFn.process() body. Derived watermark estimator
  should implement all APIs listed below. Additional methods can be implemented
  and will be available when invoked within a DoFn.

  Internal state must not be updated asynchronously.
  """

    def get_estimator_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Get current state of the WatermarkEstimator instance, which can be used\n    to recreate the WatermarkEstimator when processing the restriction. See\n    WatermarkEstimatorProvider.create_watermark_estimator.\n    '
        raise NotImplementedError(type(self))

    def current_watermark(self):
        if False:
            i = 10
            return i + 15
        'Return estimated output_watermark. This function must return\n    monotonically increasing watermarks.'
        raise NotImplementedError(type(self))

    def observe_timestamp(self, timestamp):
        if False:
            print('Hello World!')
        'Update tracking  watermark with latest output timestamp.\n\n    Args:\n      timestamp: the `timestamp.Timestamp` of current output element.\n\n    This is called with the timestamp of every element output from the DoFn.\n    '
        raise NotImplementedError(type(self))

class RestrictionProgress(object):
    """Used to record the progress of a restriction."""

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self._fraction = kwargs.pop('fraction', None)
        self._completed = kwargs.pop('completed', None)
        self._remaining = kwargs.pop('remaining', None)
        assert not kwargs

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'RestrictionProgress(fraction=%s, completed=%s, remaining=%s)' % (self._fraction, self._completed, self._remaining)

    @property
    def completed_work(self):
        if False:
            i = 10
            return i + 15
        if self._completed is not None:
            return self._completed
        elif self._remaining is not None and self._fraction is not None:
            return self._remaining * self._fraction / (1 - self._fraction)
        else:
            return self._fraction

    @property
    def remaining_work(self):
        if False:
            while True:
                i = 10
        if self._remaining is not None:
            return self._remaining
        elif self._completed is not None and self._fraction:
            return self._completed * (1 - self._fraction) / self._fraction
        else:
            return 1 - self._fraction

    @property
    def total_work(self):
        if False:
            return 10
        return self.completed_work + self.remaining_work

    @property
    def fraction_completed(self):
        if False:
            return 10
        if self._fraction is not None:
            return self._fraction
        else:
            return float(self._completed) / self.total_work

    @property
    def fraction_remaining(self):
        if False:
            print('Hello World!')
        if self._fraction is not None:
            return 1 - self._fraction
        else:
            return float(self._remaining) / self.total_work

    def with_completed(self, completed):
        if False:
            print('Hello World!')
        return RestrictionProgress(fraction=self._fraction, remaining=self._remaining, completed=completed)

class _SDFBoundedSourceRestriction(object):
    """ A restriction wraps SourceBundle and RangeTracker. """

    def __init__(self, source_bundle, range_tracker=None):
        if False:
            print('Hello World!')
        self._source_bundle = source_bundle
        self._range_tracker = range_tracker

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.__class__, (self._source_bundle,))

    def range_tracker(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._range_tracker:
            self._range_tracker = self._source_bundle.source.get_range_tracker(self._source_bundle.start_position, self._source_bundle.stop_position)
        return self._range_tracker

    def weight(self):
        if False:
            for i in range(10):
                print('nop')
        return self._source_bundle.weight

    def source(self):
        if False:
            while True:
                i = 10
        return self._source_bundle.source

    def try_split(self, fraction_of_remainder):
        if False:
            print('Hello World!')
        try:
            consumed_fraction = self.range_tracker().fraction_consumed()
            fraction = consumed_fraction + (1 - consumed_fraction) * fraction_of_remainder
            position = self.range_tracker().position_at_fraction(fraction)
            stop_pos = self._source_bundle.stop_position
            split_result = self.range_tracker().try_split(position)
            if split_result:
                (split_pos, split_fraction) = split_result
                primary_weight = self._source_bundle.weight * split_fraction
                residual_weight = self._source_bundle.weight - primary_weight
                self._source_bundle = SourceBundle(primary_weight, self._source_bundle.source, self._source_bundle.start_position, split_pos)
                return (self, _SDFBoundedSourceRestriction(SourceBundle(residual_weight, self._source_bundle.source, split_pos, stop_pos)))
        except Exception:
            return None

class _SDFBoundedSourceRestrictionTracker(RestrictionTracker):
    """An `iobase.RestrictionTracker` implementations for wrapping BoundedSource
  with SDF. The tracked restriction is a _SDFBoundedSourceRestriction, which
  wraps SourceBundle and RangeTracker.

  Delegated RangeTracker guarantees synchronization safety.
  """

    def __init__(self, restriction):
        if False:
            i = 10
            return i + 15
        if not isinstance(restriction, _SDFBoundedSourceRestriction):
            raise ValueError('Initializing SDFBoundedSourceRestrictionTracker requires a _SDFBoundedSourceRestriction. Got %s instead.' % restriction)
        self.restriction = restriction

    def current_progress(self):
        if False:
            print('Hello World!')
        return RestrictionProgress(fraction=self.restriction.range_tracker().fraction_consumed())

    def current_restriction(self):
        if False:
            while True:
                i = 10
        self.restriction.range_tracker()
        return self.restriction

    def start_pos(self):
        if False:
            for i in range(10):
                print('nop')
        return self.restriction.range_tracker().start_position()

    def stop_pos(self):
        if False:
            for i in range(10):
                print('nop')
        return self.restriction.range_tracker().stop_position()

    def try_claim(self, position):
        if False:
            return 10
        return self.restriction.range_tracker().try_claim(position)

    def try_split(self, fraction_of_remainder):
        if False:
            print('Hello World!')
        return self.restriction.try_split(fraction_of_remainder)

    def check_done(self):
        if False:
            i = 10
            return i + 15
        return self.restriction.range_tracker().fraction_consumed() >= 1.0

    def is_bounded(self):
        if False:
            i = 10
            return i + 15
        return True

class _SDFBoundedSourceWrapperRestrictionCoder(coders.Coder):

    def decode(self, value):
        if False:
            return 10
        return _SDFBoundedSourceRestriction(SourceBundle(*pickler.loads(value)))

    def encode(self, restriction):
        if False:
            for i in range(10):
                print('nop')
        return pickler.dumps((restriction._source_bundle.weight, restriction._source_bundle.source, restriction._source_bundle.start_position, restriction._source_bundle.stop_position))

class _SDFBoundedSourceRestrictionProvider(core.RestrictionProvider):
    """
  A `RestrictionProvider` that is used by SDF for `BoundedSource`.

  This restriction provider initializes restriction based on input
  element that is expected to be of BoundedSource type.
  """

    def __init__(self, desired_chunk_size=None, restriction_coder=None):
        if False:
            while True:
                i = 10
        self._desired_chunk_size = desired_chunk_size
        self._restriction_coder = restriction_coder or _SDFBoundedSourceWrapperRestrictionCoder()

    def _check_source(self, src):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(src, BoundedSource):
            raise RuntimeError('SDFBoundedSourceRestrictionProvider can only utilize BoundedSource')

    def initial_restriction(self, element_source: BoundedSource):
        if False:
            while True:
                i = 10
        self._check_source(element_source)
        range_tracker = element_source.get_range_tracker(None, None)
        return _SDFBoundedSourceRestriction(SourceBundle(None, element_source, range_tracker.start_position(), range_tracker.stop_position()))

    def create_tracker(self, restriction):
        if False:
            for i in range(10):
                print('nop')
        return _SDFBoundedSourceRestrictionTracker(restriction)

    def split(self, element, restriction):
        if False:
            return 10
        if self._desired_chunk_size is None:
            try:
                estimated_size = restriction.source().estimate_size()
            except NotImplementedError:
                estimated_size = None
            self._desired_chunk_size = Read.get_desired_chunk_size(estimated_size)
        source_bundles = restriction.source().split(self._desired_chunk_size)
        for source_bundle in source_bundles:
            yield _SDFBoundedSourceRestriction(source_bundle)

    def restriction_size(self, element, restriction):
        if False:
            while True:
                i = 10
        return restriction.weight()

    def restriction_coder(self):
        if False:
            return 10
        return self._restriction_coder

class SDFBoundedSourceReader(PTransform):
    """A ``PTransform`` that uses SDF to read from each ``BoundedSource`` in a
  PCollection.

  NOTE: This transform can only be used with beam_fn_api enabled.
  """

    def __init__(self, data_to_display=None):
        if False:
            i = 10
            return i + 15
        self._data_to_display = data_to_display or {}
        super().__init__()

    def _create_sdf_bounded_source_dofn(self):
        if False:
            print('Hello World!')

        class SDFBoundedSourceDoFn(core.DoFn):

            def __init__(self, dd):
                if False:
                    print('Hello World!')
                self._dd = dd

            def display_data(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._dd

            def process(self, unused_element, restriction_tracker=core.DoFn.RestrictionParam(_SDFBoundedSourceRestrictionProvider())):
                if False:
                    i = 10
                    return i + 15
                current_restriction = restriction_tracker.current_restriction()
                assert isinstance(current_restriction, _SDFBoundedSourceRestriction)
                return current_restriction.source().read(current_restriction.range_tracker())
        return SDFBoundedSourceDoFn(self._data_to_display)

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        return pvalue | core.ParDo(self._create_sdf_bounded_source_dofn())

    def get_windowing(self, unused_inputs):
        if False:
            return 10
        return core.Windowing(window.GlobalWindows())

    def display_data(self):
        if False:
            return 10
        return self._data_to_display