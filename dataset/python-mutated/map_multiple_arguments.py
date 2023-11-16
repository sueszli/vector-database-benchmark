def map_multiple_arguments(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam

    def strip(text, chars=None):
        if False:
            while True:
                i = 10
        return text.strip(chars)
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create(['# 🍓Strawberry\n', '# 🥕Carrot\n', '# 🍆Eggplant\n', '# 🍅Tomato\n', '# 🥔Potato\n']) | 'Strip header' >> beam.Map(strip, chars='# \n') | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    map_multiple_arguments()