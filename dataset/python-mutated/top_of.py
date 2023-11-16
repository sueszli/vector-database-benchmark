def top_of(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        shortest_elements = pipeline | 'Create produce names' >> beam.Create(['🍓 Strawberry', '🥕 Carrot', '🍏 Green apple', '🍆 Eggplant', '🌽 Corn']) | 'Shortest names' >> beam.combiners.Top.Of(2, key=len, reverse=True) | beam.Map(print)
        if test:
            test(shortest_elements)
if __name__ == '__main__':
    top_of()