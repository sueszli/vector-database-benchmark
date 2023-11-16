def tostring_element(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plant_lists = pipeline | 'Garden plants' >> beam.Create([['🍓', 'Strawberry', 'perennial'], ['🥕', 'Carrot', 'biennial'], ['🍆', 'Eggplant', 'perennial'], ['🍅', 'Tomato', 'annual'], ['🥔', 'Potato', 'perennial']]) | 'To string' >> beam.ToString.Element() | beam.Map(print)
        if test:
            test(plant_lists)
if __name__ == '__main__':
    tostring_element()