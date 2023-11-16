"""For internal use only; no backwards-compatibility guarantees."""
globals()['INT64_MAX'] = 2 ** 63 - 1
globals()['INT64_MIN'] = -2 ** 63
POWER_TEN = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0, 10000000000000.0, 100000000000000.0, 1000000000000000.0, 1e+16, 1e+17, 1e+18, 1e+19]

def get_log10_round_to_floor(element):
    if False:
        while True:
            i = 10
    power = 0
    while element >= POWER_TEN[power]:
        power += 1
    return power - 1

class DataflowDistributionCounter(object):
    """Pure python DataflowDistributionCounter in case Cython not available.


  Please avoid using python mode if possible, since it's super slow
  Cythonized DatadflowDistributionCounter defined in
  apache_beam.transforms.cy_dataflow_distribution_counter.

  Currently using special bucketing strategy suitable for Dataflow

  Attributes:
    min: minimum value of all inputs.
    max: maximum value of all inputs.
    count: total count of all inputs.
    sum: sum of all inputs.
    buckets: histogram buckets of value counts for a
    distribution(1,2,5 bucketing). Max bucket_index is 58( sys.maxint as input).
    is_cythonized: mark whether DataflowDistributionCounter cythonized.
  """
    MAX_BUCKET_SIZE = 59
    BUCKET_PER_TEN = 3

    def __init__(self):
        if False:
            return 10
        global INT64_MAX
        self.min = INT64_MAX
        self.max = 0
        self.count = 0
        self.sum = 0
        self.buckets = [0] * self.MAX_BUCKET_SIZE
        self.is_cythonized = False

    def add_input(self, element):
        if False:
            print('Hello World!')
        if element < 0:
            raise ValueError('Distribution counters support only non-negative value')
        self.min = min(self.min, element)
        self.max = max(self.max, element)
        self.count += 1
        self.sum += element
        bucket_index = self.calculate_bucket_index(element)
        self.buckets[bucket_index] += 1

    def add_input_n(self, element, n):
        if False:
            i = 10
            return i + 15
        if element < 0:
            raise ValueError('Distribution counters support only non-negative value')
        self.min = min(self.min, element)
        self.max = max(self.max, element)
        self.count += n
        self.sum += element * n
        bucket_index = self.calculate_bucket_index(element)
        self.buckets[bucket_index] += n

    def calculate_bucket_index(self, element):
        if False:
            i = 10
            return i + 15
        'Calculate the bucket index for the given element.'
        if element == 0:
            return 0
        log10_floor = get_log10_round_to_floor(element)
        power_of_ten = POWER_TEN[log10_floor]
        if element < power_of_ten * 2:
            bucket_offset = 0
        elif element < power_of_ten * 5:
            bucket_offset = 1
        else:
            bucket_offset = 2
        return 1 + log10_floor * self.BUCKET_PER_TEN + bucket_offset

    def translate_to_histogram(self, histogram):
        if False:
            print('Hello World!')
        'Translate buckets into Histogram.\n\n    Args:\n      histogram: apache_beam.runners.dataflow.internal.clents.dataflow.Histogram\n      Ideally, only call this function when reporting counter to\n      dataflow service.\n    '
        first_bucket_offset = 0
        last_bucket_offset = 0
        for index in range(0, self.MAX_BUCKET_SIZE):
            if self.buckets[index] != 0:
                first_bucket_offset = index
                break
        for index in range(self.MAX_BUCKET_SIZE - 1, -1, -1):
            if self.buckets[index] != 0:
                last_bucket_offset = index
                break
        histogram.firstBucketOffset = first_bucket_offset
        histogram.bucketCounts = self.buckets[first_bucket_offset:last_bucket_offset + 1]

    def extract_output(self):
        if False:
            while True:
                i = 10
        global INT64_MIN
        global INT64_MAX
        if not INT64_MIN <= self.sum <= INT64_MAX:
            self.sum %= 2 ** 64
            if self.sum >= INT64_MAX:
                self.sum -= 2 ** 64
        mean = self.sum // self.count if self.count else float('nan')
        return (mean, self.sum, self.count, self.min, self.max)

    def merge(self, accumulators):
        if False:
            print('Hello World!')
        raise NotImplementedError()