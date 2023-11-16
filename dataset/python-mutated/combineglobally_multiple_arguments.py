def combineglobally_multiple_arguments(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        common_items_with_exceptions = pipeline | 'Create produce' >> beam.Create([{'🍓', '🥕', '🍌', '🍅', '🌶️'}, {'🍇', '🥕', '🥝', '🍅', '🥔'}, {'🍉', '🥕', '🍆', '🍅', '🍍'}, {'🥑', '🥕', '🌽', '🍅', '🥥'}]) | 'Get common items with exceptions' >> beam.CombineGlobally(lambda sets, exclude: set.intersection(*(sets or [set()])) - exclude, exclude={'🥕'}) | beam.Map(print)
        if test:
            test(common_items_with_exceptions)
if __name__ == '__main__':
    combineglobally_multiple_arguments()