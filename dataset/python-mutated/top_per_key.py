def top_per_key(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        shortest_elements_per_key = pipeline | 'Create produce names' >> beam.Create([('spring', '🥕 Carrot'), ('spring', '🍓 Strawberry'), ('summer', '🥕 Carrot'), ('summer', '🌽 Corn'), ('summer', '🍏 Green apple'), ('fall', '🥕 Carrot'), ('fall', '🍏 Green apple'), ('winter', '🍆 Eggplant')]) | 'Shortest names per key' >> beam.combiners.Top.PerKey(2, key=len, reverse=True) | beam.Map(print)
        if test:
            test(shortest_elements_per_key)
if __name__ == '__main__':
    top_per_key()