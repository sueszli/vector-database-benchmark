def pardo_dofn_methods(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam

    class DoFnMethods(beam.DoFn):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            print('__init__')
            self.window = beam.transforms.window.GlobalWindow()

        def setup(self):
            if False:
                while True:
                    i = 10
            print('setup')

        def start_bundle(self):
            if False:
                while True:
                    i = 10
            print('start_bundle')

        def process(self, element, window=beam.DoFn.WindowParam):
            if False:
                while True:
                    i = 10
            self.window = window
            yield ('* process: ' + element)

        def finish_bundle(self):
            if False:
                i = 10
                return i + 15
            yield beam.utils.windowed_value.WindowedValue(value='* finish_bundle: 🌱🌳🌍', timestamp=0, windows=[self.window])

        def teardown(self):
            if False:
                return 10
            print('teardown')
    with beam.Pipeline() as pipeline:
        results = pipeline | 'Create inputs' >> beam.Create(['🍓', '🥕', '🍆', '🍅', '🥔']) | 'DoFn methods' >> beam.ParDo(DoFnMethods()) | beam.Map(print)
        if test:
            return test(results)
if __name__ == '__main__':
    pardo_dofn_methods()