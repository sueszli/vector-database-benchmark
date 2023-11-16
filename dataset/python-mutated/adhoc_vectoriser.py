import json

def vectorise(list):
    if False:
        print('Hello World!')
    '\n    convert list of strings into sparse (dictionary) vectors using existing terms\n    :param list: list of strings to be vectorised\n    :return: sparse vector in dictionary form\n    '
    with open('./dictionaries/term_to_id_dictionary.json') as fp:
        reference = json.load(fp)
    vector = {}
    for term in list:
        term = term.lower()
        try:
            vector[reference[term]] += 1
        except KeyError:
            try:
                vector[reference[term]] = 1
            except:
                pass
    return vector