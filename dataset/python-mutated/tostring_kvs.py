def tostring_kvs(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Garden plants' >> beam.Create([('🍓', 'Strawberry'), ('🥕', 'Carrot'), ('🍆', 'Eggplant'), ('🍅', 'Tomato'), ('🥔', 'Potato')]) | 'To string' >> beam.ToString.Kvs() | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    tostring_kvs()