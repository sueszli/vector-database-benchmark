def flatmap_function(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam

    def split_words(text):
        if False:
            i = 10
            return i + 15
        return text.split(',')
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create(['🍓Strawberry,🥕Carrot,🍆Eggplant', '🍅Tomato,🥔Potato']) | 'Split words' >> beam.FlatMap(split_words) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    flatmap_function()