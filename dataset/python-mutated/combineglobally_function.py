def combineglobally_function(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam

    def get_common_items(sets):
        if False:
            return 10
        return set.intersection(*(sets or [set()]))
    with beam.Pipeline() as pipeline:
        common_items = pipeline | 'Create produce' >> beam.Create([{'🍓', '🥕', '🍌', '🍅', '🌶️'}, {'🍇', '🥕', '🥝', '🍅', '🥔'}, {'🍉', '🥕', '🍆', '🍅', '🍍'}, {'🥑', '🥕', '🌽', '🍅', '🥥'}]) | 'Get common items' >> beam.CombineGlobally(get_common_items) | beam.Map(print)
        if test:
            test(common_items)
if __name__ == '__main__':
    combineglobally_function()