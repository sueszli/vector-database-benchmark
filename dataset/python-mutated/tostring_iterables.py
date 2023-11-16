def tostring_iterables(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants_csv = pipeline | 'Garden plants' >> beam.Create([['🍓', 'Strawberry', 'perennial'], ['🥕', 'Carrot', 'biennial'], ['🍆', 'Eggplant', 'perennial'], ['🍅', 'Tomato', 'annual'], ['🥔', 'Potato', 'perennial']]) | 'To string' >> beam.ToString.Iterables() | beam.Map(print)
        if test:
            test(plants_csv)
if __name__ == '__main__':
    tostring_iterables()