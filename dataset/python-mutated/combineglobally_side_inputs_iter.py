def combineglobally_side_inputs_iter(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        exclude = pipeline | 'Create exclude' >> beam.Create(['🥕'])
        common_items_with_exceptions = pipeline | 'Create produce' >> beam.Create([{'🍓', '🥕', '🍌', '🍅', '🌶️'}, {'🍇', '🥕', '🥝', '🍅', '🥔'}, {'🍉', '🥕', '🍆', '🍅', '🍍'}, {'🥑', '🥕', '🌽', '🍅', '🥥'}]) | 'Get common items with exceptions' >> beam.CombineGlobally(lambda sets, exclude: set.intersection(*(sets or [set()])) - set(exclude), exclude=beam.pvalue.AsIter(exclude)) | beam.Map(print)
        if test:
            test(common_items_with_exceptions)
if __name__ == '__main__':
    combineglobally_side_inputs_iter()