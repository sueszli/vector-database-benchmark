def flatmap_multiple_arguments(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam

    def split_words(text, delimiter=None):
        if False:
            print('Hello World!')
        return text.split(delimiter)
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create(['🍓Strawberry,🥕Carrot,🍆Eggplant', '🍅Tomato,🥔Potato']) | 'Split words' >> beam.FlatMap(split_words, delimiter=',') | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    flatmap_multiple_arguments()