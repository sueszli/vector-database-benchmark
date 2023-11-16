def regex_matches_kv(test=None):
    if False:
        return 10
    import apache_beam as beam
    regex = '(?P<icon>[^\\s,]+), *(\\w+), *(\\w+)'
    with beam.Pipeline() as pipeline:
        plants_matches_kv = pipeline | 'Garden plants' >> beam.Create(['🍓, Strawberry, perennial', '🥕, Carrot, biennial ignoring trailing words', '🍆, Eggplant, perennial', '🍅, Tomato, annual', '🥔, Potato, perennial', '# 🍌, invalid, format', 'invalid, 🍉, format']) | 'Parse plants' >> beam.Regex.matches_kv(regex, keyGroup='icon') | beam.Map(print)
        if test:
            test(plants_matches_kv)
if __name__ == '__main__':
    regex_matches_kv()