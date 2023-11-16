def pardo_dofn(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam

    class SplitWords(beam.DoFn):

        def __init__(self, delimiter=','):
            if False:
                print('Hello World!')
            self.delimiter = delimiter

        def process(self, text):
            if False:
                while True:
                    i = 10
            for word in text.split(self.delimiter):
                yield word
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create(['🍓Strawberry,🥕Carrot,🍆Eggplant', '🍅Tomato,🥔Potato']) | 'Split words' >> beam.ParDo(SplitWords(',')) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    pardo_dofn()