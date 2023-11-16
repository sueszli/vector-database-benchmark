def regex_replace_first(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants_replace_first = pipeline | 'Garden plants' >> beam.Create(['🍓, Strawberry, perennial', '🥕, Carrot, biennial', '🍆,\tEggplant, perennial', '🍅, Tomato, annual', '🥔, Potato, perennial']) | 'As dictionary' >> beam.Regex.replace_first('\\s*,\\s*', ': ') | beam.Map(print)
        if test:
            test(plants_replace_first)
if __name__ == '__main__':
    regex_replace_first()