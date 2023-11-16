def combineglobally_side_inputs_dict(test=None):
    if False:
        return 10
    import apache_beam as beam

    def get_custom_common_items(sets, options):
        if False:
            i = 10
            return i + 15
        sets = sets or [set()]
        common_items = set.intersection(*sets)
        common_items |= options['include']
        common_items &= options['exclude']
        return common_items
    with beam.Pipeline() as pipeline:
        options = pipeline | 'Create options' >> beam.Create([('exclude', {'🥕'}), ('include', {'🍇', '🌽'})])
        custom_common_items = pipeline | 'Create produce' >> beam.Create([{'🍓', '🥕', '🍌', '🍅', '🌶️'}, {'🍇', '🥕', '🥝', '🍅', '🥔'}, {'🍉', '🥕', '🍆', '🍅', '🍍'}, {'🥑', '🥕', '🌽', '🍅', '🥥'}]) | 'Get common items' >> beam.CombineGlobally(get_custom_common_items, options=beam.pvalue.AsDict(options)) | beam.Map(print)
        if test:
            test(custom_common_items)
if __name__ == '__main__':
    combineglobally_side_inputs_dict()