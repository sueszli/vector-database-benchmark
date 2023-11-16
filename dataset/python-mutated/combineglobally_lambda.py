def combineglobally_lambda(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        common_items = pipeline | 'Create produce' >> beam.Create([{'🍓', '🥕', '🍌', '🍅', '🌶️'}, {'🍇', '🥕', '🥝', '🍅', '🥔'}, {'🍉', '🥕', '🍆', '🍅', '🍍'}, {'🥑', '🥕', '🌽', '🍅', '🥥'}]) | 'Get common items' >> beam.CombineGlobally(lambda sets: set.intersection(*(sets or [set()]))) | beam.Map(print)
        if test:
            test(common_items)
if __name__ == '__main__':
    combineglobally_lambda()