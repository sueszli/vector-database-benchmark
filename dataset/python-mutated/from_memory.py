import apache_beam as beam

class Output(beam.PTransform):

    class _OutputFn(beam.DoFn):

        def __init__(self, prefix=''):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.prefix = prefix

        def process(self, element):
            if False:
                while True:
                    i = 10
            print(self.prefix + str(element))

    def __init__(self, label=None, prefix=''):
        if False:
            i = 10
            return i + 15
        super().__init__(label)
        self.prefix = prefix

    def expand(self, input):
        if False:
            while True:
                i = 10
        input | beam.ParDo(self._OutputFn(self.prefix))
with beam.Pipeline() as p:
    p | 'Create words' >> beam.Create(['Hello Beam', 'It`s introduction']) | 'Log words' >> Output()
    p | 'Create numbers' >> beam.Create(range(1, 11)) | 'Log numbers' >> Output()