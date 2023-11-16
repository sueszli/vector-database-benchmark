def combineglobally_side_inputs_singleton(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        single_exclude = pipeline | 'Create single_exclude' >> beam.Create(['🥕'])
        common_items_with_exceptions = pipeline | 'Create produce' >> beam.Create([{'🍓', '🥕', '🍌', '🍅', '🌶️'}, {'🍇', '🥕', '🥝', '🍅', '🥔'}, {'🍉', '🥕', '🍆', '🍅', '🍍'}, {'🥑', '🥕', '🌽', '🍅', '🥥'}]) | 'Get common items with exceptions' >> beam.CombineGlobally(lambda sets, single_exclude: set.intersection(*(sets or [set()])) - {single_exclude}, single_exclude=beam.pvalue.AsSingleton(single_exclude)) | beam.Map(print)
        if test:
            test(common_items_with_exceptions)
if __name__ == '__main__':
    combineglobally_side_inputs_singleton()