def regex_matches(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    regex = '(?P<icon>[^\\s,]+), *(\\w+), *(\\w+)'
    with beam.Pipeline() as pipeline:
        plants_matches = pipeline | 'Garden plants' >> beam.Create(['🍓, Strawberry, perennial', '🥕, Carrot, biennial ignoring trailing words', '🍆, Eggplant, perennial', '🍅, Tomato, annual', '🥔, Potato, perennial', '# 🍌, invalid, format', 'invalid, 🍉, format']) | 'Parse plants' >> beam.Regex.matches(regex) | beam.Map(print)
        if test:
            test(plants_matches)
if __name__ == '__main__':
    regex_matches()