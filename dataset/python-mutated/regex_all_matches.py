def regex_all_matches(test=None):
    if False:
        return 10
    import apache_beam as beam
    regex = '(?P<icon>[^\\s,]+), *(\\w+), *(\\w+)'
    with beam.Pipeline() as pipeline:
        plants_all_matches = pipeline | 'Garden plants' >> beam.Create(['🍓, Strawberry, perennial', '🥕, Carrot, biennial ignoring trailing words', '🍆, Eggplant, perennial', '🍅, Tomato, annual', '🥔, Potato, perennial', '# 🍌, invalid, format', 'invalid, 🍉, format']) | 'Parse plants' >> beam.Regex.all_matches(regex) | beam.Map(print)
        if test:
            test(plants_all_matches)
if __name__ == '__main__':
    regex_all_matches()