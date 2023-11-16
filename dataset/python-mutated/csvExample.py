import apache_beam as beam

class Output(beam.PTransform):

    class _OutputFn(beam.DoFn):

        def __init__(self, prefix=''):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.prefix = prefix

        def process(self, element):
            if False:
                while True:
                    i = 10
            print(self.prefix + str(element))

    def __init__(self, label=None, prefix=''):
        if False:
            return 10
        super().__init__(label)
        self.prefix = prefix

    def expand(self, input):
        if False:
            for i in range(10):
                print('nop')
        input | beam.ParDo(self._OutputFn(self.prefix))

class ExtractTaxiRideCostFn(beam.DoFn):

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        line = element.split(',')
        return tryParseTaxiRideCost(line, 16)

def tryParseTaxiRideCost(line, index):
    if False:
        return 10
    if len(line) > index:
        yield line[index]
    else:
        yield 0.0
with beam.Pipeline() as p:
    lines = p | 'Log lines' >> beam.io.ReadFromText('gs://apache-beam-samples/nyc_taxi/misc/sample1000.csv') | beam.ParDo(ExtractTaxiRideCostFn()) | beam.combiners.Sample.FixedSizeGlobally(10) | beam.FlatMap(lambda cost: cost) | Output(prefix='Taxi cost: ')